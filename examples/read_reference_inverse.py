import h5py
import numpy as np

filename = "data/reference_inverse.h5"

with h5py.File(filename, "r") as f:

    # Real parameters
    Wo = f["Wo"][:]
    V = f["V"][:]

    # Predicted Wo
    Wo_predicted = f["Wo_predicted"][:]

# ============================================================
# RANDOM CASE
# ============================================================

rng = np.random.default_rng()

iw = rng.integers(len(Wo))
iv = rng.integers(len(V))

Wo_real = Wo[iw]
Wo_fit = Wo_predicted[iw, iv]

error = (Wo_fit - Wo_real) * 100 / Wo_real

print(f"Wo = {Wo_real:.2f}")
print(f"V  = {V[iv]:.2f}")

print("")

print(f"Wo predicted = {Wo_fit:.4f}")
print(f"Error Wo     = {error:.0e} %")
