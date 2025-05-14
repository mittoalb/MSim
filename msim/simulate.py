import os
import shutil
import json
import cupy as cp
import numpy as np
import z5py
import xraylib
from scipy.interpolate import interp1d
from msim.logger import setup_custom_logger, log_exception

logger = setup_custom_logger('simulate_n5', lfname='logs/simulate_n5.log')

CONFIG_PATH = "simulate_config.json"

DEFAULT_CONFIG = {
    "DATA_N5": "/home/beams0/AMITTONE/Software/MSim/Models/Stained.n5",
    "DATA_META": "/home/beams0/AMITTONE/Software/MSim/Models/Stained.json",
    "OUTPUT_N5": "/home/beams0/AMITTONE/Software/MSim/Output/Stained_out.n5",
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

def load_config(path=CONFIG_PATH):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f"[INFO] Default config written to {path}")
        return DEFAULT_CONFIG
    with open(path, 'r') as f:
        return json.load(f)

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

def run_simulation(config_path="simulate_config.json"):
    try:
        config = load_config(config_path)
        delta_map, mu_abs_map, mu_scat_map, voxel_size = labels2map(
            config["DATA_N5"], config["DATA_META"], config["SCALE_KEY"], config["ENERGY_KEV"])

        dz = voxel_size[0] * 1e-6
        nz, ny, nx = delta_map.shape
        pad = config["PAD"]
        ny_p, nx_p = ny + 2*pad, nx + 2*pad

        delta_p = cp.pad(cp.asarray(delta_map), ((0,0),(pad,pad),(pad,pad)), mode='edge')
        mu_abs_p = cp.pad(cp.asarray(mu_abs_map), ((0,0),(pad,pad),(pad,pad)), mode='edge')
        mu_scat_p = cp.pad(cp.asarray(mu_scat_map), ((0,0),(pad,pad),(pad,pad)), mode='edge')

        DETECTOR_PIXEL = dz
        k0 = 2 * cp.pi / ((6.62607015e-34 * 2.99792458e8)/(config["ENERGY_KEV"]*1e3*1.602176634e-19))
        kx = cp.fft.fftfreq(nx_p, DETECTOR_PIXEL) * 2 * cp.pi
        ky = cp.fft.fftfreq(ny_p, DETECTOR_PIXEL) * 2 * cp.pi
        KX, KY = cp.meshgrid(kx, ky)
        H_slice = cp.exp(-1j * (KX**2 + KY**2) * dz / (2 * k0))
        H_det = cp.exp(-1j * (KX**2 + KY**2) * config["DETECTOR_DIST"] / (2 * k0))

        theta = np.linspace(0, 5e-3, 501)
        r_e = 2.8179403227e-15
        cos_t = np.cos(theta)
        dcs_r = r_e**2 * (1 + cos_t**2) / 2
        alpha = config["ENERGY_KEV"] * 1e3 / 511e3
        num = 1 + cos_t**2
        den = (1 + alpha*(1 - cos_t))**2
        dcs_c = (r_e**2 / 2) * num / den * (1 + alpha*(1 - cos_t) - alpha**2 * (1 - cos_t)**2 / (num * (1 + alpha*(1 - cos_t))))
        psf_r = (dcs_r + dcs_c) * 2 * np.pi * np.sin(theta)
        psf_r /= np.trapezoid(psf_r, theta)
        r_px = theta * config["DETECTOR_DIST"] / DETECTOR_PIXEL
        yy_g, xx_g = np.mgrid[-ny_p//2:ny_p//2, -nx_p//2:nx_p//2]
        rgrid = np.hypot(xx_g, yy_g)
        interp = interp1d(r_px, psf_r, bounds_error=False, fill_value=0)
        psf2d = interp(rgrid)
        psf2d /= psf2d.sum()
        psf2d_gpu = cp.asarray(psf2d)
        spad = psf2d.shape[0] // 2

        Psi = cp.ones((ny_p, nx_p), dtype=cp.complex64)
        wavefronts = cp.zeros((nz, ny_p, nx_p), dtype=cp.complex64)
        I_sum_z = cp.zeros((nz, ny_p, nx_p), dtype=cp.float32)

        rand_phase = cp.random.uniform(0, 2*cp.pi, size=Psi.shape, dtype=cp.float32)
        Psi *= cp.exp(1j * rand_phase * 0.001)

        for z in range(nz):
            Psi = cp.fft.ifft2(cp.fft.fft2(Psi) * H_slice)
            if config["ENABLE_PHASE"]:
                Psi *= cp.exp(1j * k0 * delta_p[z] * dz)
            if config["ENABLE_ABSORPTION"]:
                Psi *= cp.exp(-0.5 * mu_abs_p[z] * dz)
            if config["ENABLE_SCATTER"]:
                I = cp.abs(Psi)**2
                p = 1 - cp.exp(-mu_scat_p[z] * dz)
                I_sc = p * I
                I_un = (1 - p) * I
                I_pad = cp.pad(I_sc, spad, mode='edge')
                psf_pad = cp.pad(psf2d_gpu, [(0, I_pad.shape[0] - psf2d_gpu.shape[0]), (0, I_pad.shape[1] - psf2d_gpu.shape[1])])
                I_bl = cp.fft.ifft2(cp.fft.fft2(I_pad) * cp.fft.fft2(psf_pad)).real
                I_bl = I_bl[spad:-spad, spad:-spad]
                Psi = cp.sqrt(cp.maximum(I_un + I_bl, 0)) * cp.exp(1j * cp.angle(Psi))

            wavefronts[z] = Psi
            I_sum_z[z] = cp.abs(Psi)**2

        Psi = cp.fft.ifft2(cp.fft.fft2(Psi) * H_det)
        I_sim = cp.abs(Psi)**2
        I_sim = I_sim[pad:pad+ny, pad:pad+nx].get()

        wave_r = cp.real(wavefronts)[:, pad:pad+ny, pad:pad+nx].get()
        wave_i = cp.imag(wavefronts)[:, pad:pad+ny, pad:pad+nx].get()
        delta_c = delta_p[:, pad:pad+ny, pad:pad+nx].get()
        mu_abs_c = mu_abs_p[:, pad:pad+ny, pad:pad+nx].get()
        mu_scat_c = mu_scat_p[:, pad:pad+ny, pad:pad+nx].get()
        I_sum_z_c = I_sum_z[:, pad:pad+ny, pad:pad+nx].get()

        if os.path.exists(config["OUTPUT_N5"]):
            shutil.rmtree(config["OUTPUT_N5"])
        out = z5py.File(config["OUTPUT_N5"], use_zarr_format=False)
        out.create_dataset('I_sim', data=I_sim.astype('float32'), chunks=tuple(config["CHUNKS_2D"]), compression='raw')
        out.create_dataset('delta_map', data=delta_c.astype('float32'), chunks=tuple(config["CHUNKS_3D"]), compression='raw')
        out.create_dataset('mu_abs_map', data=mu_abs_c.astype('float32'), chunks=tuple(config["CHUNKS_3D"]), compression='raw')
        out.create_dataset('mu_scat_map', data=mu_scat_c.astype('float32'), chunks=tuple(config["CHUNKS_3D"]), compression='raw')
        out.create_dataset('wavefronts_real', data=wave_r.astype('float32'), chunks=tuple(config["CHUNKS_3D"]), compression='raw')
        out.create_dataset('wavefronts_imag', data=wave_i.astype('float32'), chunks=tuple(config["CHUNKS_3D"]), compression='raw')
        out.create_dataset('I_sum_z', data=I_sum_z_c.astype('float32'), chunks=tuple(config["CHUNKS_3D"]), compression='raw')

        logger.info("Saved simulation output to N5")

    except Exception as e:
        log_exception(logger, e)
