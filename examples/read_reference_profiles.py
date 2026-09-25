import h5py
import numpy as np
import matplotlib.pyplot as plt


filename = "data/reference_profiles.h5"


# ============================================================
# RANDOM CASE
# ============================================================

rng = np.random.default_rng()


with h5py.File(filename, "r") as f:

    # Select one random case
    groups = list(f.keys())
    group_name = rng.choice(groups)

    group = f[group_name]

    Wo = group.attrs["Wo"]
    V = group.attrs["V"]

    r = group["r"][:]
    z = group["z"][:]

# Shift suspension point to z = 0
z = z - z.max()


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(6, 6))

# Axisymmetric profile
plt.plot(r, z, "C0")
plt.plot(-r, z, "C0")

plt.xlabel(r"$\bar{{r}}^{{*}}$")
plt.ylabel(r"$\bar{{z}}^{{*}} - \max(\bar{{z}}^{{*}})$")
plt.title(rf"Reference profile: $\mathrm{{Wo}} = {Wo:g},\ \bar{{V}} = {V:g}$")

plt.axis("equal")
plt.tight_layout()
plt.show()
