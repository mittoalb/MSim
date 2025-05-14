#!/usr/bin/env python3
import numpy as np
import z5py
import os, shutil, json
from logger import setup_custom_logger, log_exception

logger = setup_custom_logger("chipgen", lfname="chipgen.log")

def generate_n5(
    output_dir="chip.n5",
    shape=(32,128,128),
    seed=42,
    substrate_code=1,
    logic_code=2,
    via_code=3,
    trace_code=4,
    pad_code=5,
    voxel_size=(0.25,0.25,0.1),
    chunks=(16,64,64),
    n_scales=3
):
    """
    Generate synthetic chip volume and save as an N5 multiscale container.
    Each scale is a group '0','1',... containing a dataset 'data'.
    Also stores a lookup table for material codes in JSON sidecar and N5 attributes.
    This layout is natively recognized by Fiji's N5 importer.
    """
    logger.info(f"Generating chip model, shape={shape}")
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    volume = np.zeros(shape, dtype=np.uint8)

    for z in range(2, nz, 6):
        volume[z:z+2] = substrate_code
    for z in range(nz-3):
        if z % 6 in (0,1):
            if z % 2 == 0:
                for y in range(0, ny-6, 6):
                    volume[z, y:y+2] = logic_code
            else:
                for x in range(0, nx-6, 6):
                    volume[z, :, x:x+2] = logic_code

    logger.info("Adding Cu vias...")
    for y in range(12, ny-12, 20):
        for x in range(12, nx-12, 20):
            h = rng.integers(4, nz-4)
            volume[2:h, y:y+2, x:x+2] = via_code

    for i in range(0, min(ny,nx)-10, 12):
        for z in range(2, nz-2, 4):
            for d in range(10):
                if i + d < ny and i + d < nx:
                    volume[z, i+d, i+d] = trace_code

    pad_radius, pad_spacing, thickness = 6, 20, 6
    for gy in range(8):
        for gx in range(8):
            cy, cx = 15 + gy*pad_spacing, 15 + gx*pad_spacing
            for dy in range(-pad_radius, pad_radius+1):
                for dx in range(-pad_radius, pad_radius+1):
                    if dy*dy + dx*dx <= pad_radius*pad_radius:
                        yy, xx = cy + dy, cx + dx
                        if 0 <= yy < ny and 0 <= xx < nx:
                            volume[-thickness:-1, yy, xx] = pad_code

    lookup = {
        1: {'composition': {'Si': 1.0},   'density': 2.33},
        2: {'composition': {'SiO2': 1.0}, 'density': 2.65},
        3: {'composition': {'Cu': 1.0},   'density': 8.96},
        4: {'composition': {'Al': 1.0},   'density': 2.70},
        5: {'composition': {'SnPb': 1.0}, 'density': 8.48}
    }

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    f = z5py.File(output_dir, use_zarr_format=False)
    current = volume
    for level in range(n_scales):
        grp = f.require_group(str(level))
        grp.create_dataset(
            'labels', data=current,
            chunks=chunks, dtype='uint8', compression='raw'
        )
        current = current[::2, ::2, ::2]

    f.attrs['lookup'] = lookup
    f.attrs['voxel_size'] = list(voxel_size)

    meta = {
        'voxel_size': list(voxel_size),
        'lookup': lookup,
        'description': 'Synthetic chip materials multiscale N5 container'
    }
    with open('chip.json', 'w') as out:
        json.dump(meta, out, indent=2)

    logger.info(f"Saved N5 container → {output_dir}")

if __name__ == '__main__':
    generate_n5()