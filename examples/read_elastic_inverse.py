import h5py
import numpy as np


filename = "data/elastic_inverse.h5"


with h5py.File(filename, "r") as f:

    # Real parameters
    Wo = f["Wo"][:]
    V = f["V"][:]
    K = f["K"][:]
    G = f["G"][:]
    A_frac = f["A_frac"][:]

    # Predicted K
    K_stretch = f["stretch/K_predicted"][:]
    K_area = f["area/K_predicted"][:]

# ============================================================
# RANDOM CASE
# ============================================================

rng = np.random.default_rng()

iw = rng.integers(len(Wo))
iv = rng.integers(len(V))
iK = rng.integers(len(K))
iG = rng.integers(len(G))
iA = rng.integers(len(A_frac))

idx = (iw, iv, iK, iG, iA)

# Errors
err_stretch = (K_stretch[idx] - K[iK]) * 100 / K[iK]
err_area = (K_area[idx] - K[iK]) * 100 / K[iK]

print(f"Wo     = {Wo[iw]:.2f}")
print(f"V      = {V[iv]:.2f}")
print(f"A_frac = {A_frac[iA]:.2f}")

print("")

print(f"K real = {K[iK]:.4f}")
print(f"G real = {G[iG]:.4f}")

print("")

print(f"K predicted (stretch) = {K_stretch[idx]:.4f}")
print(f"Error K (stretch)     = {err_stretch:.0e} %")

print("")

print(f"K predicted (area)    = {K_area[idx]:.4f}")
print(f"Error K (area)        = {err_area:.0e} %")
