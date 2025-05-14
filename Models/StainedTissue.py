import numpy as np
import z5py, os, shutil, json
from numba import njit
from logger import setup_custom_logger, log_exception
from scipy.ndimage import gaussian_filter

logger = setup_custom_logger("Stained_n5", lfname="Stained_n5.log")

@njit
def draw_directional_tree(mask, root, max_depth, base_radius, rng_vals):
    stack = [(root[0], root[1], root[2], 0, 0, 0.0, 1.0, 0.0, base_radius)]  # Start downward (+Y)

    while stack:
        z, y, x, depth, idx, dx, dy, dz, radius = stack.pop()
        if depth >= max_depth or idx + 100 >= len(rng_vals):
            continue

        # Normalize initial direction
        norm = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
        dx /= norm
        dy /= norm
        dz /= norm

        bounds = np.array(mask.shape)
        pz, py, px = float(z), float(y), float(x)  # Floating-point position

        direction = np.array([dz, dy, dx], dtype=np.float32)
        pos = np.array([pz, py, px], dtype=np.float32)
        t_max = 999
        for i in range(3):
            if direction[i] > 0:
                t = (bounds[i] - 1 - pos[i]) / direction[i]
            elif direction[i] < 0:
                t = -pos[i] / direction[i]
            else:
                t = np.inf
            if t < t_max:
                t_max = t

        length = int(min(t_max, 80))
        jitter_interval = 5
        end_z, end_y, end_x = int(pz), int(py), int(px)

        for i in range(length):
            if i % jitter_interval == 0:
                dx += (rng_vals[idx] - 0.5) * 1.0
                dy += (rng_vals[idx+1] - 0.5) * 1.5
                dz += (rng_vals[idx+2] - 0.5) * 1.0
                idx += 3
                norm = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
                dx /= norm
                dy /= norm
                dz /= norm

            # Advance floating-point position
            pz += dz
            py += dy
            px += dx

            zi, yi, xi = int(pz), int(py), int(px)
            end_z, end_y, end_x = zi, yi, xi

            if 0 <= zi < mask.shape[0] and 0 <= yi < mask.shape[1] and 0 <= xi < mask.shape[2]:
                for zz in range(-radius, radius + 1):
                    for yy in range(-radius, radius + 1):
                        for xx in range(-radius, radius + 1):
                            if zz**2 + yy**2 + xx**2 <= radius**2:
                                zzz, yyy, xxx = zi + zz, yi + yy, xi + xx
                                if 0 <= zzz < mask.shape[0] and 0 <= yyy < mask.shape[1] and 0 <= xxx < mask.shape[2]:
                                    mask[zzz, yyy, xxx] = 5

        # Fibonacci-based branching
        if radius > 1:
            # Fibonacci logic
            fib = [1, 1]
            while len(fib) <= depth + 2:
                fib.append(fib[-1] + fib[-2])
                num_branches = min(2 + fib[depth] % 6, 5)

            for b in range(num_branches):
                scale = 0.5 + 0.4 * rng_vals[idx + b + 3]  # range: [0.5, 0.9]
                child_radius = max(1, int(radius * scale))
                stack.append((
                    end_z, end_y, end_x, depth + 1, idx + b * 10,
                    dx + (rng_vals[idx + b] - 0.5) * 2,
                    dy + (rng_vals[idx + b + 1] - 0.5) * 2,
                    dz + (rng_vals[idx + b + 2] - 0.5) * 2,
                    child_radius
                ))


def generate_n5(
    output_dir="Stained.n5",
    n_slices=500,
    ny=500,
    nx=500,
    seed=20,
    num_cells=600,
    cell_radius_range=(2, 5),
    chunks=(64, 64, 64),
    n_scales=4,
    voxel_size=[1.0, 1.0, 1.0],
    macro_regions=4,
    region_smoothness=20,
    num_vessels=150,
    max_depth=6,
    branch_factor=5,
    vessel_radius=3
):
    try:
        logger.info(f"Generating stained tissue labels with vascular tree, shape=({n_slices},{ny},{nx})")
        rng = np.random.default_rng(seed)
        labels = np.zeros((n_slices, ny, nx), dtype=np.uint8)

        if macro_regions > 0:
            logger.info("Creating macroregions...")
            random_field = rng.random((n_slices, ny, nx))
            smooth_field = gaussian_filter(random_field, sigma=region_smoothness)
            thresholds = np.linspace(smooth_field.min(), smooth_field.max(), macro_regions + 1)
            for i in range(macro_regions):
                labels[(smooth_field >= thresholds[i]) & (smooth_field < thresholds[i + 1])] = i

        zz, yy, xx = np.ogrid[:n_slices, :ny, :nx]
        for _ in range(num_cells):
            zc, yc, xc = rng.integers(0, n_slices), rng.integers(0, ny), rng.integers(0, nx)
            rc = rng.uniform(*cell_radius_range)
            d2 = (zz - zc) ** 2 + (yy - yc) ** 2 + (xx - xc) ** 2
            labels[d2 <= rc ** 2] = 4

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
            draw_directional_tree(labels, root, max_depth=max_depth,
                                base_radius=vessel_radius, rng_vals=rng_vals)


        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        f = z5py.File(output_dir, use_zarr_format=False)
        for level in range(n_scales):
            grp = f.require_group(str(level))
            scaled = labels[::2 ** level, ::2 ** level, ::2 ** level]
            grp.create_dataset('labels', data=scaled, chunks=chunks, compression='raw')

        lookup = {
            0: {'composition': {'C': 0.60, 'H': 0.08, 'O': 0.32}, 'density': 1.18},
            1: {'composition': {'C': 0.58, 'H': 0.08, 'O': 0.32, 'Os': 0.02}, 'density': 1.20},
            2: {'composition': {'C': 0.56, 'H': 0.08, 'O': 0.32, 'Os': 0.04}, 'density': 1.22},
            3: {'composition': {'C': 0.54, 'H': 0.08, 'O': 0.32, 'Os': 0.06}, 'density': 1.24},
            4: {'composition': {'H': 0.11, 'O': 0.89}, 'density': 1.0},
            5: {'composition': {'C': 0.64, 'H': 0.06, 'O': 0.30}, 'density': 1.2},
        }

        f.attrs['lookup'] = lookup
        f.attrs['voxel_size'] = voxel_size
        with open('Stained.json', 'w') as jf:
            json.dump({
                'lookup': lookup,
                'voxel_size': voxel_size,
                'description': 'Stained tissue with macro regions, cells, and vascular network'
            }, jf, indent=2)

        logger.info(f"Saved 3D labels with vascular tree to → {output_dir}")

    except Exception as e:
        log_exception(logger, e)

if __name__ == '__main__':
    generate_n5()