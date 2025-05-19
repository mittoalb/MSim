#!/usr/bin/env python3
import os
import sys
import json
import shutil
import numpy as np
import z5py
from numba import njit, prange
from scipy.ndimage import gaussian_filter
from skimage.morphology import ball
import cupy as cp

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from logger import setup_custom_logger, log_exception

# ─────────────────────────────────────────────────────────────────────────────
# Constants
CELL_VAL    = 5
NUCLEUS_VAL = 7
AXON_VAL    = 8
VESSEL_VAL  = 5



#import numpy as np
#from carlext import carve_ball

# 'labels' is your volume: np.zeros((z,y,x), dtype=np.uint8)
# Build a list of carve events (z,y,x,radius)
#centers_list = [(z0, y0, x0, rad), ...]
#centers = np.array(centers_list, dtype=np.int32)

# Call the GPU carve
#carlext.carve_ball(labels, centers, VESSEL_VAL)


# ─────────────────────────────────────────────────────────────────────────────
# 1) Numba‐jit carving routine replaces your inner 3×3 loops


import cupy as cp

def carve_ball(mask_np, z, y, x, radius, label):
    # 1) Move the NumPy volume to GPU
    mask_gpu = cp.asarray(mask_np)

    # 2) Build the offsets & sphere mask
    d = cp.arange(-radius, radius + 1, dtype=cp.int32)
    dz, dy, dx = cp.meshgrid(d, d, d, indexing='ij')
    sphere = (dz*dz + dy*dy + dx*dx) <= (radius*radius)

    # 3) Absolute positions
    zz = z + dz; yy = y + dy; xx = x + dx

    # 4) Valid‐index mask (inside bounds & inside sphere)
    nz, ny, nx = mask_gpu.shape
    valid = (
        (zz >= 0) & (zz < nz) &
        (yy >= 0) & (yy < ny) &
        (xx >= 0) & (xx < nx) &
        sphere
    )

    # 5) Carve on the GPU buffer
    mask_gpu[zz[valid], yy[valid], xx[valid]] = label

    # 6) Copy back into your original NumPy array
    mask_np[:] = cp.asnumpy(mask_gpu)

# ─────────────────────────────────────────────────────────────────────────────
# 2) Vascular tree growth, fully in Numba
#@njit#(parallel=True)
def draw_vessels(mask, root_z, root_y, root_x, max_depth, base_radius, rng_vals):
    stack = [(root_z, root_y, root_x, 0, 0, 0.0, 0.0, 1.0, base_radius)]
    nz, ny, nx = mask.shape

    while stack:
        z, y, x, depth, idx, dx, dy, dz, radius = stack.pop()
        if depth >= max_depth or idx + 3 > len(rng_vals):
            continue

        # normalize
        norm = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6
        dx, dy, dz = dx/norm, dy/norm, dz/norm

        # compute how far we can go before leaving
        t_max = np.inf
        for i_d, d in enumerate((dz, dy, dx)):
            if d > 0:
                t = ((mask.shape[i_d]-1) - ( [z,y,x][i_d] )) / d
            elif d < 0:
                t = - ( [z,y,x][i_d] ) / d
            else:
                t = np.inf
            if t < t_max:
                t_max = t
        length = min(int(t_max), 80)

        pz, py, px = z, y, x
        for i in range(length):
            if i % 5 == 0:
                dx += (rng_vals[idx]   - 0.5) * 1.0
                dy += (rng_vals[idx+1] - 0.5) * 1.5
                dz += (rng_vals[idx+2] - 0.5) * 1.0
                idx += 3
                norm = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6
                dx, dy, dz = dx/norm, dy/norm, dz/norm

            pz += dz; py += dy; px += dx
            zi, yi, xi = int(pz), int(py), int(px)
            if 0 <= zi < nz and 0 <= yi < ny and 0 <= xi < nx:
                carve_ball(mask, zi, yi, xi, int(radius), VESSEL_VAL)

        # Fibonacci branching
        if radius > 1:
            # build enough fib numbers
            fib0, fib1 = 1, 1
            for _ in range(depth+1):
                fib0, fib1 = fib1, fib0+fib1
            num_branches = 2 + (fib1 % 6)
            if num_branches > 5:
                num_branches = 5

            for b in range(num_branches):
                base = idx + b*3
                if base + 3 > len(rng_vals):
                    break
                ndx = dx + (rng_vals[base]   - 0.5) * 2.0
                ndy = dy + (rng_vals[base+1] - 0.5) * 2.0
                ndz = dz + (rng_vals[base+2] - 0.5) * 2.0
                child_r = max(1, int(radius * (0.5 + 0.4 * rng_vals[base+2])))
                stack.append((zi, yi, xi,
                              depth+1,
                              base,
                              ndx, ndy, ndz,
                              child_r))


# ─────────────────────────────────────────────────────────────────────────────
# 3) Axon trunk + branching, also in Numba
@njit(parallel=True)
def branch_axons(mask, z, y, x, depth, idx, dx, dy, dz, radius, max_depth, rng_vals):
    stack = [(z, y, x, depth, idx, dx, dy, dz, radius)]
    nz, ny, nx = mask.shape

    while stack:
        z, y, x, depth, idx, dx, dy, dz, radius = stack.pop()
        if depth >= max_depth or idx + 3 > len(rng_vals):
            continue

        norm = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6
        dx, dy, dz = dx/norm, dy/norm, dz/norm

        # carve forward
        t_max = np.inf
        for i_d, d in enumerate((dz, dy, dx)):
            if d > 0:
                t = ((mask.shape[i_d]-1) - ( [z,y,x][i_d] )) / d
            elif d<0:
                t = - ( [z,y,x][i_d] ) / d
            else:
                t = np.inf
            if t < t_max:
                t_max = t
        length = min(int(t_max), 80)

        pz, py, px = z, y, x
        for i in range(length):
            if i % 5 == 0:
                dx += (rng_vals[idx]   - 0.5) * 1.0
                dy += (rng_vals[idx+1] - 0.5) * 1.5
                dz += (rng_vals[idx+2] - 0.5) * 1.0
                idx += 3
                norm = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6
                dx, dy, dz = dx/norm, dy/norm, dz/norm

            pz += dz; py += dy; px += dx
            zi, yi, xi = int(round(pz)), int(round(py)), int(round(px))
            if 0 <= zi < nz and 0 <= yi < ny and 0 <= xi < nx:
                carve_ball(mask, zi, yi, xi, radius, AXON_VAL)

        # Fibonacci branching
        if radius > 1:
            fib0, fib1 = 1,1
            for _ in range(depth+1):
                fib0, fib1 = fib1, fib0+fib1
            num_b = 2 + (fib1 % 6)
            if num_b > 5: num_b = 5

            for b in range(num_b):
                base = idx + b*3
                if base + 3 > len(rng_vals):
                    break
                ndx = dx + (rng_vals[base]   - 0.5) * 2.0
                ndy = dy + (rng_vals[base+1] - 0.5) * 2.0
                ndz = dz + (rng_vals[base+2] - 0.5) * 2.0
                child_r = max(1, int(radius * (0.5 + 0.4*rng_vals[base+2])))
                stack.append((zi, yi, xi,
                              depth+1,
                              base,
                              ndx, ndy, ndz,
                              child_r))


@njit(parallel=True)
def draw_axons(mask, z0, y0, x0, z1, y1, x1,
               max_depth, base_radius, rng_vals):
    # straight trunk
    dz, dy, dx = z1 - z0, y1 - y0, x1 - x0
    dist = int(np.ceil(np.sqrt(dz*dz + dy*dy + dx*dx))) + 1
    for t in range(dist):
        frac = t / (dist - 1)
        zi = int(round(z0 + frac * dz))
        yi = int(round(y0 + frac * dy))
        xi = int(round(x0 + frac * dx))
        carve_ball(mask, zi, yi, xi, base_radius, AXON_VAL)
    # branching at end
    branch_axons(mask, z1, y1, x1, 1, 0,
                 dz/dist, dy/dist, dx/dist,
                 base_radius, max_depth, rng_vals)


# ─────────────────────────────────────────────────────────────────────────────
# 4) MST‐based connectivity (Prim’s), then draw
def connect_cells(mask, centers, max_depth, axon_radius, rng_vals):
    n = len(centers)
    d2 = np.zeros((n, n), np.float64)
    for i in range(n):
        for j in range(i+1, n):
            dz = centers[j][0] - centers[i][0]
            dy = centers[j][1] - centers[i][1]
            dx = centers[j][2] - centers[i][2]
            d2[i, j] = d2[j, i] = dz*dz + dy*dy + dx*dx

    connected = {0}
    edges = []
    while len(connected) < n:
        best = (np.inf, -1, -1)
        for u in connected:
            for v in range(n):
                if v not in connected and d2[u, v] < best[0]:
                    best = (d2[u, v], u, v)
        _, u, v = best
        edges.append((u, v))
        connected.add(v)

    for u, v in edges:
        z0, y0, x0 = centers[u]
        z1, y1, x1 = centers[v]
        draw_axons(mask, z0, y0, x0, z1, y1, x1,
                   max_depth, axon_radius, rng_vals)


# ─────────────────────────────────────────────────────────────────────────────
def add_neurons(labels, voxel_size, num_cells,
                cell_radius_range, axon_dia_range, max_depth):
    nz, ny, nx = labels.shape
    dz, dy, dx = voxel_size
    px = np.mean((dz, dy, dx))
    rng = np.random.default_rng()

    zz, yy, xx = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx),
        indexing='ij'
    )

    centers = []
    for _ in range(num_cells):
        zc = rng.integers(0, nz)
        yc = rng.integers(0, ny)
        xc = rng.integers(0, nx)
        centers.append((zc, yc, xc))

        R_px = rng.uniform(*cell_radius_range) / px
        a = R_px * rng.uniform(0.6, 1.0)
        b = R_px * rng.uniform(0.6, 1.0)
        c = R_px * rng.uniform(0.6, 1.0)
        mask_cell = (((zz-zc)/c)**2 +
                     ((yy-yc)/b)**2 +
                     ((xx-xc)/a)**2) <= 1.0
        labels[mask_cell] = CELL_VAL

        r_inner = R_px * rng.uniform(0.2, 0.5)
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        off = v * (max(a, b, c) - r_inner) * rng.uniform()
        zci, yci, xci = zc + off[0], yc + off[1], xc + off[2]
        mask_inner = (((zz - zci)/r_inner)**2 +
                      ((yy - yci)/r_inner)**2 +
                      ((xx - xci)/r_inner)**2) <= 1.0
        labels[mask_inner] = NUCLEUS_VAL

    # connect via MST + axon drawing
    rng_vals = rng.random(1_000_000, dtype=np.float32)
    axon_r_px = (rng.uniform(*axon_dia_range) / 2.0) / px
    connect_cells(labels, centers, max_depth, int(axon_r_px), rng_vals)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
def add_macroregions(labels, macro_regions, region_smoothness, voxel_size):
    nz, ny, nx = labels.shape
    rng = np.random.default_rng()
    noise = rng.random((nz, nx))
    warp2d = gaussian_filter(noise, sigma=region_smoothness)
    layer_thick = ny / float(macro_regions)
    warp2d = (warp2d - warp2d.mean()) / np.ptp(warp2d) * (layer_thick * 0.4)
    warp3d = np.repeat(warp2d[:, None, :], ny, axis=1)
    yy = np.arange(ny)[None, :, None]
    base_bounds = np.linspace(0, ny, macro_regions+1)

    for i in range(macro_regions):
        y0, y1 = base_bounds[i], base_bounds[i+1]
        lower = y0 + warp3d
        upper = y1 + warp3d
        mask = (yy >= lower) & (yy < upper)
        labels[mask] = i + 1

    return labels


# ─────────────────────────────────────────────────────────────────────────────
def save_multiscale_zarr(data, codes, out_dir,
                         voxel_size, n_scales, base_chunk, logger):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    f = z5py.File(out_dir, use_zarr_format=True)
    curr = data
    datasets = []
    for lvl in range(n_scales):
        path = str(lvl)
        chunks = tuple(min(c, s) for c, s in zip(base_chunk, curr.shape))
        f.create_dataset(path, data=curr, chunks=chunks, compression="raw")
        scale = 2**lvl
        datasets.append({
            "path": path,
            "coordinateTransformations": [
                {"type": "scale",       "scale": [scale]*3},
                {"type": "translation", "translation": [scale/2-0.5]*3},
            ]
        })
        curr = curr[::2, ::2, ::2]
    f.attrs["lookup"]     = {int(k):v for k,v in codes.items()}
    f.attrs["voxel_size"] = voxel_size
    multiscale_meta = {
        "version": "0.4",
        "axes": [
            {"name":"z","type":"space"},
            {"name":"y","type":"space"},
            {"name":"x","type":"space"},
        ],
        "datasets": datasets,
        "type": "image",
        "metadata": {"voxel_size": voxel_size},
    }
    f.attrs["multiscales"] = [multiscale_meta]
    with open(os.path.join(out_dir, "multiscale.json"), "w") as fj:
        json.dump([multiscale_meta], fj, indent=2)
    if logger:
        logger.info(f"Saved multiscale Zarr → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
def generate_brain(config_path="sim_config.json"):
    logger = setup_custom_logger("brain_sample", lfname="brain_sample.log")
    try:
        with open(config_path,'r') as cf:
            cfg = json.load(cf)
        params = cfg["generate_brain"]
        lookup = cfg["materials"]
        dims = (params["n_slices"], params["ny"], params["nx"])
        labels = np.zeros(dims, dtype=np.uint8)

        if params["macro_regions"]>0:
            labels = add_macroregions(
                labels,
                params["macro_regions"],
                params["region_smoothness"],
                params["voxel_size"]
            )
            logger.info(f"Applied {params['macro_regions']} macro regions.")

        logger.info(f"Adding {params['num_cells']} neurons...")
        labels = add_neurons(
            labels,
            params["voxel_size"],
            num_cells=params["num_cells"],
            cell_radius_range=(2,10),
            axon_dia_range=(1,2),
            max_depth=params["max_depth"]
        )

        logger.info(f"Adding {params['num_vessels']} vessels...")
        rng = np.random.default_rng(params["seed"])
        for _ in range(params["num_vessels"]):
            face = rng.integers(0,6)
            if face==0:
                root=(0, rng.integers(0,params["ny"]), rng.integers(0,params["nx"]))
            elif face==1:
                root=(params["n_slices"]-1, rng.integers(0,params["ny"]), rng.integers(0,params["nx"]))
            elif face==2:
                root=(rng.integers(0,params["n_slices"]), 0, rng.integers(0,params["nx"]))
            elif face==3:
                root=(rng.integers(0,params["n_slices"]), params["ny"]-1, rng.integers(0,params["nx"]))
            elif face==4:
                root=(rng.integers(0,params["n_slices"]), rng.integers(0,params["ny"]), 0)
            else:
                root=(rng.integers(0,params["n_slices"]), rng.integers(0,params["ny"]), params["nx"]-1)
            rng_vals = rng.random(50000, dtype=np.float32)
            draw_vessels(
                labels,
                root[0], root[1], root[2],
                params["max_depth"],
                params["vessel_radius"],
                rng_vals
            )

        save_multiscale_zarr(
            data=labels,
            codes=lookup,
            out_dir=params["output_dir"],
            voxel_size=tuple(params["voxel_size"]),
            n_scales=params["n_scales"],
            base_chunk=tuple(params["chunks"]),
            logger=logger
        )

    except Exception as e:
        log_exception(logger, e)


if __name__ == "__main__":
    generate_brain()

