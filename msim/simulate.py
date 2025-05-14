import os
import shutil
import json
import numpy as np
import z5py
import xraylib
from numpy.fft import fft2, ifft2
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from msim.logger import setup_custom_logger, log_exception

# --- Logger setup -----------------------------------------------------------
logger = setup_custom_logger('simulate_n5', lfname='logs/simulate_n5.log')

# --- Default config + path --------------------------------------------------
CONFIG_PATH = "simulate_config.json"

DEFAULT_CONFIG = {
    "DATA_N5": "Models/Stained.n5",
    "DATA_META": "Models/Stained.json",
    "OUTPUT_N5": "Output/Stained_out.n5",
    "SCALE_KEY": "0",
    "CHUNKS_3D": [64, 64, 64],
    "CHUNKS_2D": [256, 256],
    "PAD": 50,
    "ENERGY_KEV": 23.0,
    "DETECTOR_DIST": 0.3,
    "ENABLE_PHASE": True,
    "ENABLE_ABSORPTION": True,
    "ENABLE_SCATTER": True
}

# --- Load or create config file --------------------------------------------
def load_config(path=CONFIG_PATH):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f"[INFO] Default config written to {path}")
        return DEFAULT_CONFIG
    with open(path, 'r') as f:
        return json.load(f)

# --- Load label-based maps --------------------------------------------------
def labels2map(n5_path, json_path, scale_key, energy_kev):
    with open(json_path, 'r') as f:
        meta = json.load(f)
    voxel_size = meta.get("voxel_size")
    logger.info(f"Read voxel size: {voxel_size}")

    lookup = meta["lookup"] if "lookup" in meta else meta

    fz = z5py.File(n5_path, use_zarr_format=False)
    labels = fz[scale_key]['labels'][...]

    nz, ny, nx = labels.shape
    delta = np.zeros((nz, ny, nx), dtype='float32')
    mu_abs = np.zeros((nz, ny, nx), dtype='float32')
    mu_scat = np.zeros((nz, ny, nx), dtype='float32')

    xraylib.XRayInit()
    E = energy_kev

    for k, props in lookup.items():
        mask = labels == int(k)
        formula = ''.join([f"{el}{amt}" for el, amt in props['composition'].items()])
        rho = props['density']

        delta_val = 1 - xraylib.Refractive_Index_Re(formula, E, rho)
        cs_tot = xraylib.CS_Total_CP(formula, E)
        cs_rayl = xraylib.CS_Rayl_CP(formula, E)

        delta[mask] = delta_val
        mu_abs[mask] = rho * (cs_tot - cs_rayl) * 100
        mu_scat[mask] = rho * cs_rayl * 100

    return delta, mu_abs, mu_scat, tuple(voxel_size)

# --- Main simulation wrapper ------------------------------------------------
def run_simulation(config_path="simulate_config.json"):
    try:
        config = load_config(config_path)

        DATA_N5          = config["DATA_N5"]
        DATA_META        = config["DATA_META"]
        OUTPUT_N5        = config["OUTPUT_N5"]
        SCALE_KEY        = config["SCALE_KEY"]
        CHUNKS_3D        = tuple(config["CHUNKS_3D"])
        CHUNKS_2D        = tuple(config["CHUNKS_2D"])
        PAD              = config["PAD"]
        ENERGY_KEV       = config["ENERGY_KEV"]
        DETECTOR_DIST    = config["DETECTOR_DIST"]
        ENABLE_PHASE     = config["ENABLE_PHASE"]
        ENABLE_ABSORPTION= config["ENABLE_ABSORPTION"]
        ENABLE_SCATTER   = config["ENABLE_SCATTER"]

        delta_map, mu_abs_map, mu_scat_map, voxel_size = labels2map(DATA_N5, DATA_META, SCALE_KEY, ENERGY_KEV)
        dz = voxel_size[0] * 1e-6
        nz, ny, nx = delta_map.shape
        DETECTOR_PIXEL = dz

        pad = PAD
        delta_p = np.pad(delta_map, ((0,0),(pad,pad),(pad,pad)), 'edge')
        mu_abs_p = np.pad(mu_abs_map, ((0,0),(pad,pad),(pad,pad)), 'edge')
        mu_scat_p = np.pad(mu_scat_map, ((0,0),(pad,pad),(pad,pad)), 'edge')
        ny_p, nx_p = ny + 2*pad, nx + 2*pad

        k0 = 2*np.pi / ((6.62607015e-34*2.99792458e8)/(ENERGY_KEV*1e3*1.602176634e-19))
        kx = np.fft.fftfreq(nx_p, DETECTOR_PIXEL)*2*np.pi
        ky = np.fft.fftfreq(ny_p, DETECTOR_PIXEL)*2*np.pi
        KX, KY = np.meshgrid(kx, ky)
        H_slice = np.exp(-1j*(KX**2+KY**2)*dz/(2*k0))
        H_det   = np.exp(-1j*(KX**2+KY**2)*DETECTOR_DIST/(2*k0))

        r_e = 2.8179403227e-15
        theta = np.linspace(0, 5e-3, 501)
        cos_t = np.cos(theta)
        dcs_r = r_e**2 * (1 + cos_t**2) / 2
        alpha = ENERGY_KEV * 1e3 / 511e3
        num = 1 + cos_t**2
        den = (1 + alpha*(1 - cos_t))**2
        dcs_c = (r_e**2 / 2) * num / den * (1 + alpha*(1 - cos_t) - alpha**2 * (1 - cos_t)**2 / (num * (1 + alpha*(1 - cos_t))))
        psf_r = (dcs_r + dcs_c) * 2 * np.pi * np.sin(theta)
        psf_r /= np.trapezoid(psf_r, theta)
        r_px = theta * DETECTOR_DIST / DETECTOR_PIXEL
        yy_g, xx_g = np.mgrid[-ny_p//2:ny_p//2, -nx_p//2:nx_p//2]
        rgrid = np.hypot(xx_g, yy_g)
        interp = interp1d(r_px, psf_r, bounds_error=False, fill_value=0)
        psf2d = interp(rgrid)
        psf2d /= psf2d.sum()
        spad = psf2d.shape[0] // 2

        Psi = np.ones((ny_p, nx_p), dtype='complex64')
        wavefronts = np.zeros((nz, ny_p, nx_p), dtype='complex64')
        I_sum_z = np.zeros((nz, ny_p, nx_p), dtype='float32')

        phase_noise = np.exp(1j * np.random.uniform(0, 2*np.pi, size=Psi.shape) * 0.001)
        Psi *= phase_noise

        for z in range(nz):
            Psi = ifft2(fft2(Psi) * H_slice)
            if ENABLE_PHASE:
                Psi *= np.exp(1j * k0 * delta_p[z] * dz)
            if ENABLE_ABSORPTION:
                Psi *= np.exp(-0.5 * mu_abs_p[z] * dz)
            if ENABLE_SCATTER:
                I = np.abs(Psi)**2
                p = 1 - np.exp(-mu_scat_p[z] * dz)
                I_sc = p * I
                I_un = (1 - p) * I
                I_pad = np.pad(I_sc, spad, 'edge')
                I_bl = fftconvolve(I_pad, psf2d, 'same')[spad:-spad, spad:-spad]
                Psi = np.sqrt(np.maximum(I_un + I_bl, 0)) * np.exp(1j * np.angle(Psi))

            wavefronts[z] = Psi.copy()
            I_sum_z[z] = np.abs(Psi)**2
            logger.info(f"z={z:3d} |\u03a8|^2 sum: {np.sum(np.abs(Psi)**2):.4e}")

        Psi = ifft2(fft2(Psi) * H_det)
        I_sim = np.abs(Psi)**2
        I_sim = I_sim[pad:pad+ny, pad:pad+nx]

        delta_c = delta_p[:, pad:pad+ny, pad:pad+nx]
        mu_abs_c = mu_abs_p[:, pad:pad+ny, pad:pad+nx]
        mu_scat_c = mu_scat_p[:, pad:pad+ny, pad:pad+nx]
        wave_r = np.real(wavefronts)[:, pad:pad+ny, pad:pad+nx]
        wave_i = np.imag(wavefronts)[:, pad:pad+ny, pad:pad+nx]
        I_sum_z_c = I_sum_z[:, pad:pad+ny, pad:pad+nx]

        if os.path.exists(OUTPUT_N5):
            shutil.rmtree(OUTPUT_N5)
        out = z5py.File(OUTPUT_N5, use_zarr_format=False)
        out.create_dataset('I_sim', data=I_sim.astype('float32'), chunks=CHUNKS_2D, compression='raw')
        out.create_dataset('delta_map', data=delta_c.astype('float32'), chunks=CHUNKS_3D, compression='raw')
        out.create_dataset('mu_abs_map', data=mu_abs_c.astype('float32'), chunks=CHUNKS_3D, compression='raw')
        out.create_dataset('mu_scat_map', data=mu_scat_c.astype('float32'), chunks=CHUNKS_3D, compression='raw')
        out.create_dataset('wavefronts_real', data=wave_r.astype('float32'), chunks=CHUNKS_3D, compression='raw')
        out.create_dataset('wavefronts_imag', data=wave_i.astype('float32'), chunks=CHUNKS_3D, compression='raw')
        out.create_dataset('I_sum_z', data=I_sum_z_c.astype('float32'), chunks=CHUNKS_3D, compression='raw')

        logger.info("Saved simulation output to N5")

    except Exception as e:
        log_exception(logger, e)
