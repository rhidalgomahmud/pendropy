import h5py
import numpy as np
import matplotlib.pyplot as plt


elastic_filename = "data/elastic_profiles.h5"
reference_filename = "data/reference_profiles.h5"


# ============================================================
# RANDOM CASE
# ============================================================

rng = np.random.default_rng()


with h5py.File(elastic_filename, "r") as f:

    # Select one random case
    groups = list(f.keys())
    group_name = rng.choice(groups)

    group = f[group_name]

    Wo = group.attrs["Wo"]
    V = group.attrs["V"]
    K = group.attrs["K"]
    G = group.attrs["G"]
    A_frac = group.attrs["A_frac"]

    r_elastic = group["r"][:]
    z_elastic = group["z"][:]


with h5py.File(reference_filename, "r") as f:

    # Read corresponding reference profile
    reference_group = f[f"Wo_{Wo:g}_V_{V:g}"]

    r_reference = reference_group["r"][:]
    z_reference = reference_group["z"][:]


# Shift suspension point to z = 0
z_reference = z_reference - z_reference.max()
z_elastic = z_elastic - z_elastic.max()


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(6, 6))

# Reference profile
plt.plot(r_reference, z_reference, "C0--", label="Reference")
plt.plot(-r_reference, z_reference, "C0--")

# Elastic profile
plt.plot(r_elastic, z_elastic, "C1", label="Elastic")
plt.plot(-r_elastic, z_elastic, "C1")

plt.xlabel(r"$\bar{r}^{*}$")
plt.ylabel(r"$\bar{z}^{*} - \max(\bar{z}^{*})$")

plt.title(
    rf"Elastic profile: $\mathrm{{Wo}} = {Wo:g},\ \bar{{V}} = {V:g}$"
    "\n"
    rf"$\bar{{K}} = {K:g},\ \bar{{G}} = {G:g},\ A_{{\mathrm{{def}}}} = {A_frac:g}$"
)

plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()
