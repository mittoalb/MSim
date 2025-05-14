import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
import xraylib

def projection(volume_labels, lookup, voxel_size, config):
    """
    Simulate detector intensity from a rotated label volume.
    
    Parameters:
        volume_labels : (nz, ny, nx) integer labels (rotated volume)
        lookup        : dict mapping label to {composition, density}
        voxel_size    : (dz, dy, dx) in microns
        config        : dict containing ENERGY_KEV, DETECTOR_DIST, PAD, etc.

    Returns:
        I_sim : (ny, nx) simulated detector intensity
    """
    dz, dy, dx = [v * 1e-6 for v in voxel_size]
    nz, ny, nx = volume_labels.shape
    pad = config["PAD"]
    energy = config["ENERGY_KEV"]
    det_dist = config["DETECTOR_DIST"]
    pixel_size = dy  # assuming square pixels, lateral voxel size

    delta = np.zeros_like(volume_labels, dtype='float32')
    mu_abs = np.zeros_like(volume_labels, dtype='float32')
    mu_scat = np.zeros_like(volume_labels, dtype='float32')

    xraylib.XRayInit()
    for k, props in lookup.items():
        mask = volume_labels == int(k)
        formula = ''.join([f"{el}{amt}" for el, amt in props['composition'].items()])
        rho = props['density']
        delta_val = 1 - xraylib.Refractive_Index_Re(formula, energy, rho)
        cs_tot = xraylib.CS_Total_CP(formula, energy)
        cs_rayl = xraylib.CS_Rayl_CP(formula, energy)
        delta[mask] = delta_val
        mu_abs[mask] = rho * (cs_tot - cs_rayl) * 100
        mu_scat[mask] = rho * cs_rayl * 100

    # Padding
    delta_p = np.pad(delta, ((0, 0), (pad, pad), (pad, pad)), 'edge')
    mu_abs_p = np.pad(mu_abs, ((0, 0), (pad, pad), (pad, pad)), 'edge')
    mu_scat_p = np.pad(mu_scat, ((0, 0), (pad, pad), (pad, pad)), 'edge')
    ny_p, nx_p = ny + 2 * pad, nx + 2 * pad

    # FFT propagation kernel
    k0 = 2 * np.pi / ((6.62607015e-34 * 2.99792458e8) / (energy * 1e3 * 1.602176634e-19))
    kx = np.fft.fftfreq(nx_p, pixel_size) * 2 * np.pi
    ky = np.fft.fftfreq(ny_p, pixel_size) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    H_slice = np.exp(-1j * (KX**2 + KY**2) * dz / (2 * k0))
    H_det = np.exp(-1j * (KX**2 + KY**2) * det_dist / (2 * k0))

    # Scattering PSF
    r_e = 2.8179403227e-15
    theta = np.linspace(0, 5e-3, 501)
    cos_t = np.cos(theta)
    dcs_r = r_e**2 * (1 + cos_t**2) / 2
    alpha = energy * 1e3 / 511e3
    num = 1 + cos_t**2
    den = (1 + alpha * (1 - cos_t))**2
    dcs_c = (r_e**2 / 2) * num / den * (
        1 + alpha * (1 - cos_t) - alpha**2 * (1 - cos_t)**2 / (num * (1 + alpha * (1 - cos_t)))
    )
    psf_r = (dcs_r + dcs_c) * 2 * np.pi * np.sin(theta)
    psf_r /= np.trapezoid(psf_r, theta)
    r_px = theta * det_dist / pixel_size
    yy, xx = np.mgrid[-ny_p // 2:ny_p // 2, -nx_p // 2:nx_p // 2]
    rgrid = np.hypot(xx, yy)
    psf2d = interp1d(r_px, psf_r, bounds_error=False, fill_value=0)(rgrid)
    psf2d /= psf2d.sum()
    spad = psf2d.shape[0] // 2

    # Propagate wavefront
    Psi = np.ones((ny_p, nx_p), dtype='complex64') * np.exp(1j * np.random.uniform(0, 2*np.pi, (ny_p, nx_p)) * 0.001)
    for z in range(nz):
        Psi = ifft2(fft2(Psi) * H_slice)
        if config["ENABLE_PHASE"]:
            Psi *= np.exp(1j * k0 * delta_p[z] * dz)
        if config["ENABLE_ABSORPTION"]:
            Psi *= np.exp(-0.5 * mu_abs_p[z] * dz)
        if config["ENABLE_SCATTER"]:
            I = np.abs(Psi)**2
            p = 1 - np.exp(-mu_scat_p[z] * dz)
            I_sc = p * I
            I_un = (1 - p) * I
            I_bl = fftconvolve(np.pad(I_sc, spad, 'edge'), psf2d, mode='same')[spad:-spad, spad:-spad]
            Psi = np.sqrt(np.maximum(I_un + I_bl, 0)) * np.exp(1j * np.angle(Psi))

    # Detector propagation
    Psi = ifft2(fft2(Psi) * H_det)
    I_sim = np.abs(Psi)**2
    return I_sim[pad:pad + ny, pad:pad + nx]
