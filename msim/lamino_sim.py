import json
import numpy as np
import h5py
import z5py

from phys_sim import projection
from LSim_wrap import rotate_volume, build_quaternion
from logger import setup_custom_logger, log_exception

# Setup logger
logger = setup_custom_logger("lamino_sim", lfname="../logs/lamino_sim.log")

try:
    # --- Load lamino scan settings ---
    LAMINO_CONFIG_PATH = "../lamino_config.json"
    with open(LAMINO_CONFIG_PATH, "r") as f:
        lamino_cfg = json.load(f)

    angle_min, angle_max = lamino_cfg["ANGLES"]
    n_proj     = lamino_cfg["N_PROJ"]
    TILT_DEG   = lamino_cfg["TILT_DEG"]
    OUTPUT_H5  = lamino_cfg["OUTPUT_H5"]
    INCLUDE_WHITE = lamino_cfg.get("INCLUDE_WHITE", True)

    ANGLES = np.linspace(angle_min, angle_max, num=n_proj)
    TILT_RAD = np.deg2rad(TILT_DEG)

    # --- Load simulation parameters ---
    SIM_CONFIG_PATH = "../simulate_config.json"
    with open(SIM_CONFIG_PATH, "r") as f:
        config = json.load(f)

    DATA_N5   = config["DATA_N5"]
    META_JSON = config["DATA_META"]
    SCALE_KEY = config["SCALE_KEY"]

    logger.info(f"Using volume from: {DATA_N5}")

    # --- Load label volume and metadata ---
    def load_labels(n5_path, scale_key):
        f = z5py.File(n5_path, use_zarr_format=False)
        return f[scale_key]['labels'][...]

    def load_lookup(json_path):
        with open(json_path, 'r') as f:
            meta = json.load(f)
        voxel_size = meta.get("voxel_size")
        lookup = meta["lookup"] if "lookup" in meta else meta
        return voxel_size, lookup

    labels = load_labels(DATA_N5, SCALE_KEY)
    voxel_size, lookup = load_lookup(META_JSON)

    nz, ny, nx = labels.shape
    rotated = np.empty_like(labels, dtype=labels.dtype)
    projections = []

    # --- Simulate white field (optional) ---
    white = None
    if INCLUDE_WHITE:
        logger.info("[INFO] Simulating white field (flat illumination)...")
        white = projection(np.zeros_like(labels, dtype=labels.dtype), lookup, voxel_size, config)

    # --- Loop over rotation angles and simulate projections ---
    for angle_deg in ANGLES:
        theta_rad = np.deg2rad(angle_deg)
        print('here')
        quat = build_quaternion(TILT_RAD, theta_rad)
        print('here')
        labels = np.ascontiguousarray(labels, dtype=np.float32)
        rotated = np.ascontiguousarray(rotated, dtype=np.float32)
        assert labels.shape == rotated.shape
        assert labels.ndim == 3

        rotate_volume(labels, rotated, quat)
        print('here')

        I_sim = projection(rotated, lookup, voxel_size, config)
        print('here')
    
        projections.append(I_sim)
        logger.info(f"Angle {angle_deg:+.2f}° done")

    stack = np.stack(projections, axis=0)  # (n_proj, ny, nx)

    # --- Save output to HDF5 ---
    with h5py.File(OUTPUT_H5, 'w') as f:
        exch = f.create_group("exchange")
        exch.create_dataset("data", data=stack.astype('float32'), compression='gzip')
        if white is not None:
            exch.create_dataset("white", data=white[np.newaxis, ...].astype('float32'), compression='gzip')
        f.create_dataset("angles", data=ANGLES.astype('float32'))

    logger.info(f"[OK] Saved simulated laminography scan to: {OUTPUT_H5}")

except Exception as e:
    log_exception(logger, e)
