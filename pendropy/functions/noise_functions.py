import numpy as np

def gaussian(var_def, n_pixels, seed, roi_width_pixels=1024):

    _, r, z = var_def
    
    rng = np.random.default_rng(seed=seed)

    sigma = n_pixels * 3.0 / roi_width_pixels

    r_noise = r + rng.normal(0.0, sigma, size=r.shape)
    z_noise = z + rng.normal(0.0, sigma, size=z.shape)

    return np.zeros_like(r), r_noise, z_noise


def calibration(var_def, n_pixels, roi_width_pixels=1024):

    _, r, z = var_def

    eps = n_pixels * 3.0 / roi_width_pixels

    r_scaled = (1.0 + eps) * r
    z_scaled = (1.0 + eps) * z

    return np.zeros_like(r), r_scaled, z_scaled


def offset(var_def, n_pixels, roi_width_pixels=1024):

    _, r, z = var_def

    delta_r = n_pixels * 3.0 / roi_width_pixels

    r_offset = r + delta_r
    z_offset = z.copy()

    return np.zeros_like(r), r_offset, z_offset


def rotation(var_def, n_pixels, roi_width_pixels=1024):

    _, r, z = var_def

    delta = n_pixels * 3.0 / roi_width_pixels

    theta = np.arctan(delta)
    theta_deg = np.degrees(theta)
    print(theta_deg)

    r_rot = r * np.cos(theta) - z * np.sin(theta)
    z_rot = r * np.sin(theta) + z * np.cos(theta)

    return np.zeros_like(r), r_rot, z_rot
