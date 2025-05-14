#!/usr/bin/env python3
"""
foam_n5.py

Generate a 3D foam label volume and save it as an N5 container,
including a lookup table and voxel size for downstream processing.

Dependencies:
  pip install z5py numpy
"""
import numpy as np
import z5py
import os
import shutil
import json
from logger import setup_custom_logger

# initialize logger
logger = setup_custom_logger("foamgen", lfname="foamgen.log")

def generate_foam_n5(
    output_dir="foam.n5",
    shape=(100,200,200),    # (nz, ny, nx)
    num_bubbles=1,
    seed=42,
    voxel_size=(1.0,1.0,1.0),  # physical size (z,y,x)
    chunks=(64,64,64)         # chunk sizes for N5
):
    """
    Create a 3D foam:
      0 → water (H2O)
      1 → PMMA monomer (C5H8O2)
    and write as an N5 container with one scale group '0'.
    """
    logger.info(f"Generating foam: shape={shape}, bubbles={num_bubbles}")
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape

    # initialize volume with monomer
    volume = np.ones(shape, dtype=np.uint8)
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]

    # carve water bubbles
    for _ in range(num_bubbles):
        zc = rng.integers(0, nz)
        yc = rng.integers(0, ny)
        xc = rng.integers(0, nx)
        r  = rng.integers(10, min(nz, ny, nx)//4)
        dist = np.sqrt((zz-zc)**2 + (yy-yc)**2 + (xx-xc)**2)
        volume[dist <= r] = 0

    # material lookup
    lookup = {
        0: ("H2O", 1.0),      # water density = 1.0 g/cm3
        1: ("C5H8O2", 1.19)   # PMMA monomer ~1.19 g/cm3
    }

    # remove existing store
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # create N5 container
    f = z5py.File(output_dir, use_zarr_format=False)
    grp = f.require_group("0")
    grp.create_dataset(
        "foam",
        data=volume,
        chunks=chunks,
        dtype="uint8",
        compression="raw"
    )

    # attach metadata
    f.attrs["lookup"] = {str(k): list(v) for k,v in lookup.items()}
    f.attrs["voxel_size"] = list(voxel_size)

    # write sidecar JSON
    meta = {
        "voxel_size": list(voxel_size),
        "lookup": {str(k): list(v) for k,v in lookup.items()},
        "description": "3D foam label volume"
    }
    with open("foam_n5.json", "w") as out:
        json.dump(meta, out, indent=2)

    logger.info(f"Saved foam N5 → {output_dir}")

if __name__ == '__main__':
    generate_foam_n5()