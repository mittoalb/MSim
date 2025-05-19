#!/usr/bin/env python3
import os
import sys
import json
import shutil
import numpy as np
import z5py
import math

# make sure our project root is on PYTHONPATH
current_dir = os.path.dirname(__file__)
parent_dir  = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from logger import setup_custom_logger, log_exception
import brain
from utils import save_mzarr

# ─────────────────────────────────────────────────────────────────────────────
# Constants
CELL_VAL    = 5
NUCLEUS_VAL = 7
AXON_VAL    = 8
VESSEL_VAL  = 5

# ─────────────────────────────────────────────────────────────────────────────
def generate_brain(config_path="sim_config.json"):
    logger = setup_custom_logger("brain_sample", lfname="brain_sample.log")
    try:
        with open(config_path, "r") as cf:
            cfg = json.load(cf)
        params = cfg["generate_brain"]
        lookup = cfg["materials"]

        # Allocate the label volume and an occupancy mask (uint8!)
        dims     = (params["n_slices"], params["ny"], params["nx"])
        labels   = np.zeros(dims, dtype=np.uint8)
        occupied = np.zeros(dims, dtype=np.uint8)

        # ────────────── Macro‐regions ──────────────
        mr = params.get("macro_regions", 0)
        if mr > 0:
            brain.add_macroregions(
                labels, occupied,
                mr, params["region_smoothness"]
            )
            logger.info(f"Applied {mr} macro regions.")
        else:
            logger.info("Skipping macro-regions (macro_regions=0)")

        # ────────────── Neurons & Axons ──────────────
        num_cells = params.get("num_cells", 0)
        if num_cells > 0:
            logger.info(f"Adding {num_cells} neurons…")
            total_neuron_length = brain.add_neurons(
                labels, occupied,
                tuple(params["voxel_size"]),
                num_cells,
                tuple(params["cell_radius_range"]),
                tuple(params["axon_dia_range"]),
                params["max_depth"]
            )
            logger.info(f"Total neuron length: {total_neuron_length:.1f} voxels")
        else:
            logger.info("Skipping neuron placement (num_cells=0)")

        # ────────────── Vessels ──────────────
        num_vessels             = params.get("num_vessels", 0)
        vessel_radius_avg       = params["vessel_radius_avg"]
        vessel_radius_jitter    = params.get("vessel_radius_jitter", 0.0)
        max_depth               = params["max_depth"]
        trunk_len               = params.get("vessel_trunk_len",       150)
        jitter_interval         = params.get("vessel_jitter_interval", 15)
        max_branches            = params.get("vessel_max_branches",    2)
        branch_len              = params.get("vessel_branch_len",      trunk_len)
        radius_decay            = params.get("vessel_radius_decay",    0.8)
        seed                    = params["seed"]

        if num_vessels > 0:
            logger.info(f"Adding {num_vessels} vessels with avg radius {vessel_radius_avg} ±{vessel_radius_jitter*100:.0f}%…")
            total_vessel_length = brain.add_vessels(
                labels,
                num_vessels,
                max_depth,
                vessel_radius_avg,
                vessel_radius_jitter,
                trunk_len,
                jitter_interval,
                max_branches,
                branch_len,
                radius_decay,
                seed
            )
            logger.info(f"Total vessel center-line length: {total_vessel_length:.1f} voxels")
        else:
            logger.info("Skipping vessel growth (num_vessels=0)")

        # ────────────── Save Multiscale Zarr ──────────────
        save_mzarr(
            data       = labels,
            codes      = lookup,
            out_dir    = params["output_dir"],
            voxel_size = tuple(params["voxel_size"]),
            n_scales   = params["n_scales"],
            base_chunk = tuple(params["chunks"]),
            logger     = logger
        )

    except Exception as e:
        log_exception(logger, e)

if __name__ == "__main__":
    generate_brain()

