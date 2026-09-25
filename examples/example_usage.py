import numpy as np
import matplotlib.pyplot as plt

import pendropy.functions.pendant_drop_functions as drop
import pendropy.functions.noise_functions as noise

# ============================================================
# PARAMETERS
# ============================================================

Wo = 0.6
V = 20.0

K = 2.0
G = 2.0
A_def = 0.15

# ============================================================
# REFERENCE FORWARD
# ============================================================

print('\nReference forward')

reference_forward = drop.reference_forward(Wo=Wo, V=V, verbose=2)

par_ref = reference_forward["par_ref"]
var_ref = reference_forward["var_ref"]

r_ref = var_ref[1]
z_ref = var_ref[2]

# ============================================================
# REFERENCE INVERSE
# ============================================================

print('\nReference inverse')

reference_inverse = drop.reference_inverse(V=V, var_ref=var_ref, verbose=2)

Wo_fit = reference_inverse["fit"]

error_Wo = (Wo_fit - Wo) * 100 / Wo

# ============================================================
# ELASTIC FORWARD
# ============================================================

print('\nElastic forward')

elastic_forward = drop.elastic_forward(Wo=Wo, par_ref=par_ref, var_ref=var_ref, mod_def=(K, G), A_def=A_def, verbose=2)

var_def = elastic_forward["var_def"]

r_def = var_def[1]
z_def = var_def[2]

# ============================================================
# NOISE ADDITION
# ============================================================

var_def_noise = noise.gaussian(var_def=var_def, n_pixels=1, seed=0)

r_def_noise = var_def_noise[1]
z_def_noise = var_def_noise[2]

# ============================================================
# ELASTIC INVERSE - STRETCH
# ============================================================

print('\nElastic inverse - Stretch based')

elastic_inverse_stretch = drop.elastic_inverse_stretch(Wo=Wo_fit, V=V, var_def=var_def_noise, verbose=2)

K_stretch = elastic_inverse_stretch["fit"][0]

error_K_stretch = (K_stretch - K) * 100 / K

# ============================================================
# ELASTIC INVERSE - AREA
# ============================================================

print('\nElastic inverse - Area based')

elastic_inverse_area = drop.elastic_inverse_area(Wo=Wo_fit, V=V, var_def=var_def_noise, verbose=2)

K_area = elastic_inverse_area["fit"][0]

error_K_area = (K_area - K) * 100 / K

# ============================================================
# RESULTS
# ============================================================

print("REFERENCE INVERSE")
print("-----------------")
print(f"Wo real      = {Wo:.4f}")
print(f"Wo predicted = {Wo_fit:.4f}")
print(f"Error Wo     = {error_Wo:.0e} %")

print("")

print("ELASTIC INVERSE")
print("---------------")
print(f"K real = {K:.4f}")
print(f"G real = {G:.4f}")

print("")

print(f"K predicted (stretch) = {K_stretch:.4f}")
print(f"Error K (stretch)     = {error_K_stretch:.0e} %")

print("")

print(f"K predicted (area)    = {K_area:.4f}")
print(f"Error K (area)        = {error_K_area:.0e} %")

# ============================================================
# PLOT
# ============================================================

z_ref = z_ref - z_ref.max()
z_def = z_def - z_def.max()
z_def_noise = z_def_noise - z_def_noise.max()

plt.figure(figsize=(6, 6))

# Reference profile
plt.plot(r_ref, z_ref, "C0--", label="Reference")
plt.plot(-r_ref, z_ref, "C0--")

# Elastic profile
plt.plot(r_def, z_def, "C1", label="Elastic")
plt.plot(-r_def, z_def, "C1")

# Noisy profile

plt.plot(r_def, z_def, ".C2", ms=5, label="Noisy")
plt.plot(-r_def, z_def, ".C2")

plt.xlabel(r"$\bar{r}^{*}$")
plt.ylabel(r"$\bar{z}^{*} - \max(\bar{z}^{*})$")

plt.title(
    rf"$\mathrm{{Wo}} = {Wo:g},\ \bar{{V}} = {V:g}$"
    "\n"
    rf"$\bar{{K}} = {K:g},\ \bar{{G}} = {G:g},\ "
    rf"A_{{\mathrm{{def}}}} = {A_def:g}$"
)

plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()
