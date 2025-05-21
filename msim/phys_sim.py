import numpy as np
import cupy as cp
import cupyx.scipy.signal as cpxsignal
from scipy.interpolate import interp1d
import xraylib
import cProfile
import pstats

def projection(volume_labels, lookup, voxel_size, config):
    """
    Simulate detector intensity from a rotated label volume—GPU version.
    
    Parameters:
        volume_labels : (nz, ny, nx) integer labels (NumPy array)
        lookup        : dict mapping label to {composition, density}
        voxel_size    : (dz, dy, dx) in microns (tuple of floats)
        config        : dict containing ENERGY_KEV, DETECTOR_DIST, PAD, ENABLE_*, etc.

    Returns:
        I_sim : (ny, nx) simulated detector intensity (NumPy array)
    """
    # start profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # --- 1) Compute δ, μ_abs, μ_scat on CPU ---
    nz, ny, nx = volume_labels.shape
    delta_cpu   = cp.zeros_like(volume_labels, dtype='float32')
    mu_abs_cpu  = cp.zeros_like(volume_labels, dtype='float32')
    mu_scat_cpu = cp.zeros_like(volume_labels, dtype='float32')

    xraylib.XRayInit()
    energy = config["ENERGY_KEV"]
    for k, props in lookup.items():        
        mask    = (volume_labels == int(k))
        formula = ''.join(f"{el}{amt}" for el, amt in props['composition'].items())
        rho     = props['density']

        δ_val   = 1 - xraylib.Refractive_Index_Re(formula, energy, rho)
        cs_tot  = xraylib.CS_Total_CP(formula, energy)
        cs_rayl = xraylib.CS_Rayl_CP(formula, energy)

        delta_cpu[  mask] = δ_val
        mu_abs_cpu[ mask] = rho * (cs_tot - cs_rayl) * 100
        mu_scat_cpu[mask] = rho * cs_rayl * 100

    # --- 2) Transfer to GPU and pad ---
    pad = config["PAD"]
    delta_p   = cp.pad(delta_cpu,   ((0,0),(pad,pad),(pad,pad)), mode='edge')
    mu_abs_p  = cp.pad(mu_abs_cpu,  ((0,0),(pad,pad),(pad,pad)), mode='edge')
    mu_scat_p = cp.pad(mu_scat_cpu, ((0,0),(pad,pad),(pad,pad)), mode='edge')
    ny_p, nx_p = ny + 2*pad, nx + 2*pad

    # voxel sizes (m)
    dz, dy, dx = [v * 1e-6 for v in voxel_size]
    pixel_size = dy

    # --- 3) Build propagation kernels on GPU ---
    # wavenumber
    k0 = 2 * cp.pi / ((6.62607015e-34 * 2.99792458e8) / (energy*1e3*1.602176634e-19))
    kx = cp.fft.fftfreq(nx_p, pixel_size) * 2 * cp.pi
    ky = cp.fft.fftfreq(ny_p, pixel_size) * 2 * cp.pi
    KX, KY = cp.meshgrid(kx, ky)
    H_slice = cp.exp(-1j * (KX**2 + KY**2) * dz / (2*k0))
    H_det   = cp.exp(-1j * (KX**2 + KY**2) * config["DETECTOR_DIST"] / (2*k0))

    # --- 4) Build scattering PSF on CPU, then transfer ---
    r_e   = 2.8179403227e-15
    theta = np.linspace(0, 5e-3, 501)
    cos_t = np.cos(theta)
    dcs_r = r_e**2 * (1 + cos_t**2) / 2
    alpha = energy*1e3/511e3
    num   = 1 + cos_t**2
    den   = (1 + alpha*(1 - cos_t))**2
    dcs_c = (r_e**2/2) * num/den * (1 + alpha*(1 - cos_t)
            - alpha**2*(1 - cos_t)**2/(num*(1 + alpha*(1 - cos_t))))
    psf_r = (dcs_r + dcs_c) * 2 * np.pi * np.sin(theta)
    psf_r /= np.trapz(psf_r, theta)

    r_px = theta * config["DETECTOR_DIST"] / pixel_size
    yy, xx = np.mgrid[-ny_p//2:ny_p//2, -nx_p//2:nx_p//2]
    rgrid = np.hypot(xx, yy)
    psf2d = interp1d(r_px, psf_r, bounds_error=False, fill_value=0)(rgrid)
    psf2d /= psf2d.sum()
    psf2d_gpu = cp.asarray(psf2d, dtype=cp.float32)

    # --- 5) Initialize wavefront on GPU ---
    Psi = cp.ones((ny_p, nx_p), dtype=cp.complex64)
    rand_phase = cp.random.uniform(0, 2*cp.pi, (ny_p, nx_p), dtype=cp.float32)
    Psi *= cp.exp(1j * rand_phase * 0.001)

    # --- 6) Propagate slice-by-slice ---
    for z in range(nz):
        # free-space to next slice
        Psi = cp.fft.ifft2(cp.fft.fft2(Psi) * H_slice)

        if config["ENABLE_PHASE"]:
            Psi *= cp.exp(1j * k0 * delta_p[z] * dz)
        if config["ENABLE_ABSORPTION"]:
            Psi *= cp.exp(-0.5 * mu_abs_p[z] * dz)
        if config["ENABLE_SCATTER"]:
            I    = cp.abs(Psi)**2
            p    = 1 - cp.exp(-mu_scat_p[z] * dz)
            I_sc = p * I
            I_un = (1 - p) * I
            # FFT-based convolution on GPU
            I_bl = cpxsignal.fftconvolve(I_sc, psf2d_gpu, mode='same')
            Psi  = cp.sqrt(cp.maximum(I_un + I_bl, 0)) * cp.exp(1j*cp.angle(Psi))

    # --- 7) Final propagation to detector ---
    Psi   = cp.fft.ifft2(cp.fft.fft2(Psi) * H_det)
    I_sim = cp.abs(Psi)**2

    # stop profiler
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    #stats.print_stats(20)
    stats.dump_stats('simulate_profile.prof')
    #logger.info("Wrote profile to simulate_profile.prof")

    # return cropped detector image back on CPU
    return cp.asnumpy(I_sim[pad:pad+ny, pad:pad+nx])
