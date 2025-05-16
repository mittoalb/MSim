import numpy as np
import z5py, os, shutil, json
from numba import njit
import sys
from MSim.iodata import save_multiscale_zarr


current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from logger import setup_custom_logger, log_exception

from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

rng = np.random.default_rng()


logger = setup_custom_logger("brain_sample", lfname="brain_sample.log")

CELL_VAL    = 5
NUCLEUS_VAL = 7
AXON_VAL    = 8

@njit
def draw_vessels(mask, root, max_depth, base_radius, rng_vals):
    """
    Grow vascular branches from `root` with jitter + Fibonacci branching,
    safely indexing into rng_vals.
    """
    stack = [(root[0], root[1], root[2], 0, 0, 0.0, 1.0, 0.0, base_radius)]

    while stack:
        z, y, x, depth, idx, dx, dy, dz, radius = stack.pop()
        # only consume 3 random values per step
        if depth >= max_depth or idx + 3 > len(rng_vals):
            continue

        # Normalize direction
        norm = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6
        dx, dy, dz = dx/norm, dy/norm, dz/norm

        # compute travel limit
        bounds = np.array(mask.shape, dtype=np.float32)
        pos = np.array([float(z), float(y), float(x)], dtype=np.float32)
        direction = np.array([dz, dy, dx], dtype=np.float32)
        t_max = np.inf
        for i_d in range(3):
            d = direction[i_d]
            if d > 0:
                t = (bounds[i_d] - 1 - pos[i_d]) / d
            elif d < 0:
                t = -pos[i_d] / d
            else:
                t = np.inf
            if t < t_max:
                t_max = t

        length = int(min(t_max, 80))
        jitter_interval = 5
        end_z, end_y, end_x = int(pos[0]), int(pos[1]), int(pos[2])

        # advance and carve vessel
        pz, py, px = pos
        for i in range(length):
            if i % jitter_interval == 0:
                dx += (rng_vals[idx]   - 0.5) * 1.0
                dy += (rng_vals[idx+1] - 0.5) * 1.5
                dz += (rng_vals[idx+2] - 0.5) * 1.0
                idx += 3
                norm = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6
                dx, dy, dz = dx/norm, dy/norm, dz/norm

            pz += dz; py += dy; px += dx
            zi, yi, xi = int(pz), int(py), int(px)
            end_z, end_y, end_x = zi, yi, xi

            if 0 <= zi < mask.shape[0] and 0 <= yi < mask.shape[1] and 0 <= xi < mask.shape[2]:
                rz = int(radius)
                for zz in range(-rz, rz+1):
                    for yy in range(-rz, rz+1):
                        for xx in range(-rz, rz+1):
                            if zz*zz + yy*yy + xx*xx <= radius*radius:
                                zzz, yyy, xxx = zi+zz, yi+yy, xi+xx
                                if (0 <= zzz < mask.shape[0] and
                                    0 <= yyy < mask.shape[1] and
                                    0 <= xxx < mask.shape[2]):
                                    mask[zzz, yyy, xxx] = 5

        # Fibonacci-based branching
        if radius > 1:
            fib = [1, 1]
            while len(fib) <= depth + 2:
                fib.append(fib[-1] + fib[-2])
            num_branches = min(2 + fib[depth] % 6, 5)

            for b in range(num_branches):
                base = idx + b * 3
                if base + 3 > len(rng_vals):
                    break
                # perturb direction
                ndx = dx + (rng_vals[base]   - 0.5) * 2.0
                ndy = dy + (rng_vals[base+1] - 0.5) * 2.0
                ndz = dz + (rng_vals[base+2] - 0.5) * 2.0
                # compute child radius
                scale = 0.5 + 0.4 * rng_vals[base+2]
                child_radius = max(1, int(radius * scale))
                stack.append((
                    end_z, end_y, end_x,
                    depth + 1,
                    base,
                    ndx, ndy, ndz,
                    child_radius
                ))

def branch_axons(mask, root, current_depth, max_depth,
                 dx, dy, dz, base_radius, rng_vals, rng_idx):
    """
    Recursively (or via explicit stack) grow jitter+Fibonacci branches
    from `root` using direction (dx,dy,dz).
    """
    nz, ny, nx = mask.shape
    stack = [(root[0], root[1], root[2],
              current_depth, rng_idx, dx, dy, dz, base_radius)]

    while stack:
        z, y, x, depth, idx, dx, dy, dz, radius = stack.pop()
        if depth >= max_depth or idx+3 > len(rng_vals):
            continue

        # Normalize direction
        norm = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6
        dx, dy, dz = dx/norm, dy/norm, dz/norm

        # Carve forward up to 80 voxels
        bounds = np.array(mask.shape, np.float32)
        pos = np.array([z, y, x], np.float32)
        direction = np.array([dz, dy, dx], np.float32)
        t_max = np.inf
        for i, d in enumerate(direction):
            if d>0:
                t = (bounds[i]-1 - pos[i])/d
            elif d<0:
                t = -pos[i]/d
            else:
                t = np.inf
            t_max = min(t_max, t)
        length = int(min(t_max, 80))

        pz, py, px = pos
        jitter = 5
        endpoint = (z, y, x)

        for i in range(length):
            if i % jitter == 0:
                dx += (rng_vals[idx]   - 0.5) * 1.0
                dy += (rng_vals[idx+1] - 0.5) * 1.5
                dz += (rng_vals[idx+2] - 0.5) * 1.0
                idx += 3
                norm = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6
                dx, dy, dz = dx/norm, dy/norm, dz/norm

            pz += dz; py += dy; px += dx
            zi, yi, xi = int(round(pz)), int(round(py)), int(round(px))
            endpoint = (zi, yi, xi)

            if 0 <= zi < nz and 0 <= yi < ny and 0 <= xi < nx:
                rz = int(radius)
                for dz0 in range(-rz, rz+1):
                    for dy0 in range(-rz, rz+1):
                        for dx0 in range(-rz, rz+1):
                            if dz0*dz0 + dy0*dy0 + dx0*dx0 <= radius*radius:
                                zzz = zi+dz0; yyy = yi+dy0; xxx = xi+dx0
                                if (0<=zzz<nz and 0<=yyy<ny and 0<=xxx<nx):
                                    mask[zzz,yyy,xxx] = AXON_VAL

        # Fibonacci branching
        if radius > 1:
            fib = [1,1]
            while len(fib) <= depth+2:
                fib.append(fib[-1] + fib[-2])
            nbranches = min(2 + fib[depth] % 6, 5)
            for b in range(nbranches):
                if idx + (b+1)*3 > len(rng_vals):
                    break
                scale = 0.5 + 0.4 * rng_vals[idx + b + 3]
                child_r = max(1, int(radius * scale))
                # new direction perturbed from parent dx,dy,dz
                ndx = dx + (rng_vals[idx+b]   - 0.5) * 2
                ndy = dy + (rng_vals[idx+b+1] - 0.5) * 2
                ndz = dz + (rng_vals[idx+b+2] - 0.5) * 2

                stack.append((
                    endpoint[0], endpoint[1], endpoint[2],
                    depth+1,
                    idx + b*3,
                    ndx, ndy, ndz,
                    child_r
                ))

def draw_axons(mask, root, target, max_depth, base_radius, rng_vals):
    """
    1) Draw a straight cylinder (radius=base_radius) from root → target.
    2) At the exact target voxel, invoke branching via `branch_axons`.
    """
    # 1) straight trunk
    z0, y0, x0 = root
    z1, y1, x1 = target
    dz, dy, dx = z1 - z0, y1 - y0, x1 - x0
    dist = int(np.ceil(np.sqrt(dz*dz + dy*dy + dx*dx))) + 1

    for t in np.linspace(0, 1, dist):
        zt = z0 + t * dz
        yt = y0 + t * dy
        xt = x0 + t * dx
        zi, yi, xi = int(round(zt)), int(round(yt)), int(round(xt))
        rz = int(base_radius)
        for dz0 in range(-rz, rz+1):
            for dy0 in range(-rz, rz+1):
                for dx0 in range(-rz, rz+1):
                    if dz0*dz0 + dy0*dy0 + dx0*dx0 <= base_radius*base_radius:
                        zzz = zi + dz0
                        yyy = yi + dy0
                        xxx = xi + dx0
                        if (0 <= zzz < mask.shape[0] and
                            0 <= yyy < mask.shape[1] and
                            0 <= xxx < mask.shape[2]):
                            mask[zzz, yyy, xxx] = AXON_VAL

    # 2) now sprout the jittery/Fibonacci branches starting from the endpoint
    branch_axons(
        mask,
        root=(z1, y1, x1),
        current_depth=1,
        max_depth=max_depth,
        dx=(dx/dist), dy=(dy/dist), dz=(dz/dist),  # use unit trunk direction
        base_radius=base_radius,
        rng_vals=rng_vals,
        rng_idx=0
    )

def connect_cells(labels, centers, max_depth, axon_rad_px, rng_vals):
    """
    Build an MST over centers, then for each edge (u→v)
    call draw_axon_trunk_then_branch so that every trunk
    actually links the two cell bodies.
    """
    n = len(centers)
    # build MST (Prim)
    connected = {0}
    edges = []
    d2 = np.zeros((n,n), np.float64)
    for i in range(n):
        for j in range(i+1, n):
            dz = centers[j][0]-centers[i][0]
            dy = centers[j][1]-centers[i][1]
            dx = centers[j][2]-centers[i][2]
            d2[i,j] = d2[j,i] = dz*dz + dy*dy + dx*dx

    while len(connected) < n:
        best = (np.inf, -1, -1)
        for u in connected:
            for v in range(n):
                if v not in connected and d2[u,v] < best[0]:
                    best = (d2[u,v], u, v)
        _, u, v = best
        edges.append((u,v))
        connected.add(v)

    # draw each connected trunk+branches
    for u,v in edges:
        draw_axons(
            labels,
            root=centers[u],
            target=centers[v],
            max_depth=max_depth,
            base_radius=axon_rad_px,
            rng_vals=rng_vals
        )

def add_neurons(labels, voxel_size, num_cells=50,
                cell_radius_range=(1,10),
                axon_dia_range=(1,2),
                max_depth=5):
    nz, ny, nx = labels.shape
    dz, dy, dx = voxel_size
    px = np.mean([dz,dy,dx])
    rng = np.random.default_rng()

    # grid for spheroids
    zz, yy, xx = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx),
        indexing='ij'
    )

    centers = []
    # 1) place cell bodies + nuclei
    for _ in range(num_cells):
        zc,yc,xc = rng.integers([0,0,0],[nz,ny,nx])
        centers.append((int(zc),int(yc),int(xc)))

        R_px = rng.uniform(*cell_radius_range) / px
        a = R_px*rng.uniform(0.6,1.0)
        b = R_px*rng.uniform(0.6,1.0)
        c = R_px*rng.uniform(0.6,1.0)
        mask_cell = (((zz-zc)/c)**2 + ((yy-yc)/b)**2 + ((xx-xc)/a)**2) <= 1.0
        labels[mask_cell] = CELL_VAL

        r_inner = R_px*rng.uniform(0.2,0.5)
        v = rng.normal(size=3); v/=np.linalg.norm(v)
        off = v*(max(a,b,c)-r_inner)*rng.uniform()
        zci,yci,xci = zc+off[0], yc+off[1], xc+off[2]
        mask_inner = (((zz-zci)/r_inner)**2 +
                      ((yy-yci)/r_inner)**2 +
                      ((xx-xci)/r_inner)**2) <= 1.0
        labels[mask_inner] = NUCLEUS_VAL

    # 2) grow axons connecting each pair
    rng_vals = rng.random(1_000_000, dtype=np.float32)
    axon_r_px = (rng.uniform(*axon_dia_range)/2.0)/px
    connect_cells(labels, centers, max_depth, axon_r_px, rng_vals)

    return labels

def add_macroregions(labels, macro_regions, region_smoothness, voxel_size):
    """
    Replace your existing gaussian‐threshold code with this:
     - Build N horizontal “layers” of equal nominal thickness
     - Warp the boundaries in X (and optionally Z) with smooth noise
     - Assign each voxel to its warped layer
    """
    nz, ny, nx = labels.shape

    # 1) Generate a 2D warp‐field for each slice
    #    shape = (nz, nx), i.e. variation along X per Z
    rng = np.random.default_rng()
    noise = rng.random((nz, nx))
    warp2d = gaussian_filter(noise, sigma=region_smoothness)  
    # normalize to ±½ layer‐thickness
    layer_thick = ny / float(macro_regions)
    warp2d = (warp2d - warp2d.mean()) / np.ptp(warp2d) * (layer_thick * 0.4)

    # 2) Expand warp into a full (z,y,x) field
    #    so that each slice has the same warp profile across Y
    warp3d = np.repeat(warp2d[:, None, :], ny, axis=1)  # now (z,y,x)

    # 3) Define straight, unwarped layer boundaries
    #    e.g. layer 0 covers Y in [0,   layer_thick),
    #         layer 1 covers Y in [layer_thick, 2*layer_thick), etc.
    yy = np.arange(ny)[None, :, None]
    base_bounds = np.linspace(0, ny, macro_regions+1)

    # 4) Assign each voxel to whichever warped layer it falls into
    for i in range(macro_regions):
        y0 = base_bounds[i]
        y1 = base_bounds[i+1]
        # lower and upper boundaries, warped per (z,x)
        lower = y0 + warp3d
        upper = y1 + warp3d
        mask = (yy >= lower) & (yy < upper)
        labels[mask] = i+1   # or whatever label you like

    return labels


def generate_brain(config_path="sim_config.json"):
    """
    Generate a 3D label volume with macro regions, neurons, and vascular trees.
    All parameters and material lookup (composition & density) are loaded from a combined JSON config.
    """
    try:
        # Load configuration
        with open(config_path, 'r') as cfgf:
            cfg = json.load(cfgf)
        params = cfg.get('generate_brain', {})
        lookup = cfg.get('materials', {})

        # Unpack parameters
        output_dir       = params['output_dir']
        n_slices         = params['n_slices']
        ny               = params['ny']
        nx               = params['nx']
        seed             = params['seed']
        num_cells        = params['num_cells']
        chunks           = tuple(params['chunks'])
        n_scales         = params['n_scales']
        voxel_size       = params['voxel_size']
        macro_regions    = params['macro_regions']
        region_smoothness= params['region_smoothness']
        num_vessels      = params['num_vessels']
        max_depth        = params['max_depth']
        vessel_radius    = params['vessel_radius']

        logger.info(
            f"Generating stained tissue labels with vascular tree, shape=({n_slices},{ny},{nx})"
        )
        rng = np.random.default_rng(seed)
        labels = np.zeros((n_slices, ny, nx), dtype=np.uint8)

        # Apply macro-regions
        if macro_regions > 0:
            labels = add_macroregions(
                labels,
                macro_regions,
                region_smoothness,
                voxel_size
            )
            logger.info(f"Applied {macro_regions} warped macro-regions.")

        # Add neurons
        logger.info(f"Generating {num_cells} neurons...")
        labels = add_neurons(
            labels,
            voxel_size,
            num_cells=num_cells,
            cell_radius_range=(2, 10),
            axon_dia_range=(1, 2),
            max_depth=max_depth
        )

        # Add vascular trees
        logger.info(f"Generating {num_vessels} vascular trees...")
        for _ in range(num_vessels):
            face = rng.integers(0, 6)
            if face == 0:
                root = (0, rng.integers(0, ny), rng.integers(0, nx))
            elif face == 1:
                root = (n_slices - 1, rng.integers(0, ny), rng.integers(0, nx))
            elif face == 2:
                root = (rng.integers(0, n_slices), 0, rng.integers(0, nx))
            elif face == 3:
                root = (rng.integers(0, n_slices), ny - 1, rng.integers(0, nx))
            elif face == 4:
                root = (rng.integers(0, n_slices), rng.integers(0, ny), 0)
            else:
                root = (rng.integers(0, n_slices), rng.integers(0, ny), nx - 1)

            rng_vals = rng.random(50000)
            draw_vessels(
                labels,
                root,
                max_depth=max_depth,
                base_radius=vessel_radius,
                rng_vals=rng_vals
            )

        # Save using modular Zarr writer
        save_labeled_multiscale_zarr(
            labels      = labels,
            codes       = lookup,
            output_dir  = output_dir,
            voxel_size  = tuple(voxel_size),
            n_scales    = n_scales,
            base_chunk  = chunks,
            config      = params,
            logger      = logger
        )

    except Exception as e:
        log_exception(logger, e)



if __name__ == '__main__':
    generate_brain()