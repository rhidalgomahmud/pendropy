# -*- coding: utf-8 -*-

import numpy as np
import scipy as sc

# ------------------------------------------------------------
# REFERENCE DROP (FORWARD)
# ------------------------------------------------------------

def reference_forward(Wo, V, N_mesh=100, for_tol=1e-5, N_it=3, N_div=2, f_subdiv=5, V_exp=2, verbose=0):

    # Mesh
    eps = 1e-5
    s = np.linspace(eps, 1.0, N_mesh)

    # For returning last converged solution
    last_converged_sol = {}

    # Differential equations
    def ODEs(s, y, par, V_it):
        l, P0 = par
        psi, r, z, v = y

        dpsi_ds = l * (P0 - np.sin(psi) / r - (2.0 * np.pi * Wo / V_it) * z)
        dr_ds = l * np.cos(psi)
        dz_ds = l * np.sin(psi)
        dv_ds = np.pi * r**2 * dz_ds

        return np.vstack((dpsi_ds, dr_ds, dz_ds, dv_ds))

    # Iterative continuation in volume
    def forward_iterative(N_it, N_div, f_subdiv):
        it = 0
        
        while it < N_it:

            # Volume mesh
            V_frac_array = 1.0 / N_div + (1.0 - 1.0 / N_div) * (1.0 - (1.0 - np.linspace(0.0, 1.0, N_div))**V_exp)

            # Initial guesses
            V_first = V * V_frac_array[0]

            l_init = ((3.0 * V_first) / (4.0 * np.pi))**(1.0 / 3.0)
            P0_init = 2.0 / l_init

            par_est = np.array([l_init, P0_init], dtype=float)

            psi_init = (s * l_init * P0_init) / 2.0
            r_init = s * l_init
            z_init = (s**2 * l_init**2 * P0_init) / 4.0
            v_init = np.pi * r_init**2 * z_init

            var_est = np.vstack((psi_init, r_init, z_init, v_init))

            if verbose > 0:
                print(f"\nIteration {it + 1}/{N_it}")

            convergence = True

            for i, V_frac_it in enumerate(V_frac_array):

                V_it = V * V_frac_it

                if verbose > 1:
                    print(f"  [{i + 1}/{len(V_frac_array)}] V = {V_it}")

                # Boundary conditions
                def BCs(ya, yb, par):
                    l, P0 = par

                    res = np.zeros(6)
                    res[0] = ya[0] - (s[0] * l * P0) / 2.0
                    res[1] = ya[1] - s[0] * l
                    res[2] = ya[2] - (s[0]**2 * l**2 * P0) / 4.0
                    res[3] = yb[1] - 1.0
                    res[4] = ya[3] - (np.pi * l**4 * P0 / 8.0) * s[0]**4
                    res[5] = yb[3] - V_it

                    return res

                try:
                    sol = sc.integrate.solve_bvp(fun=lambda s, y, par: ODEs(s, y, par, V_it), bc=BCs, x=s, y=var_est, p=par_est, tol=for_tol, max_nodes=10000, verbose=0)

                    if sol.success:
                        var_est = sol.sol(s)
                        par_est = sol.p

                        # Store last converged solution
                        last_converged_sol["par_ref"] = np.array([par_est[0], par_est[1]])
                        last_converged_sol["var_ref"] = np.array([var_est[0], var_est[1], var_est[2], var_est[3]])
                        last_converged_sol["V_frac"] = V_frac_it
                        last_converged_sol["V"] = V_it

                    else:
                        convergence = False

                        if verbose > 0:
                            print(f"\nFailed to converge at V_frac = {V_frac_it}")
                            print(f"V_it = {V_it}")
                            print(f"Message: {sol.message}")

                        break

                except RuntimeError as e:
                    convergence = False

                    if verbose > 0:
                        print(f"\nFailed to converge at V_frac = {V_frac_it}")
                        print(f"V_it = {V_it}")
                        print(f"Message: {str(e)}")

                    break

            if convergence:
                if verbose > 0:
                    print("\nBVP converged")
                return par_est, var_est

            N_div *= f_subdiv
            it += 1

        raise RuntimeError("nonconvergent BVP")

    # Run solver
    try:
        par_best, var_best = forward_iterative(N_it=N_it, N_div=N_div, f_subdiv=f_subdiv)

        l_ref, P0_ref = par_best
        psi_ref, r_ref, z_ref, v_ref = var_best

        return {"success": True, "message": "success", "par_ref": np.array([l_ref, P0_ref]), "var_ref": np.array([psi_ref, r_ref, z_ref, v_ref])}
        
    except RuntimeError:
        print("\nERROR: nonconvergent BVP")

        if last_converged_sol:
            return {"success": False, "message": "nonconvergent", "par_ref": last_converged_sol["par_ref"], "var_ref": last_converged_sol["var_ref"]}

        return {"success": False, "message": "nonconvergent", "par_ref": np.array([np.nan, np.nan]), "var_ref": np.full((4, len(s)), np.nan)}
    
# ------------------------------------------------------------
# REFERENCE DROP (INVERSE)
# ------------------------------------------------------------

def reference_inverse(V, var_ref, inv_tol=1e-8, verbose=0):

    # Reference drop
    _, r_ref, z_ref, _ = var_ref

    # Mesh
    N_mesh = len(r_ref)

    # Fit penalty
    penalty = 1e6 * np.ones(2 * N_mesh)

    # Residuals
    def residuals(x):

        Wo = x[0]
        
        try:

            result_try = reference_forward(Wo=Wo, V=V, N_mesh=N_mesh)
            var_try = result_try["var_ref"]
            _, r_try, z_try, _ = var_try

        except Exception:
            return penalty

        # Scaling
        scale_r = np.max(np.abs(r_ref))
        scale_z = np.max(np.abs(z_ref))

        res_r = (r_try - r_ref) / scale_r
        res_z = (z_try - z_ref) / scale_z

        res = np.concatenate((res_r, res_z))
        
        return res
        
    # Initial guess
    Wo0 = 0.1
    x0 = np.array([Wo0], dtype=float)
    
    # Bounds
    lb = np.array([1e-3], dtype=float)
    ub = np.array([1.0], dtype=float)

    ajuste = sc.optimize.least_squares(
        residuals,
        x0=x0,
        bounds=(lb, ub),
        ftol=inv_tol,
        xtol=inv_tol,
        gtol=inv_tol,
        max_nfev=30,
        verbose=verbose
    )

    Wo_aj = ajuste.x[0]

    if verbose > 0:

        print("Ajuste final:")
        print(f"Wo = {Wo_aj:.8f}")
        print(f"least_squares success = {ajuste.success}")
        print(f"nfev = {ajuste.nfev}")

    return {
        "success": bool(ajuste.success),
        "message": ajuste.message,
        "fit": Wo_aj,
    }

# ------------------------------------------------------------
# ELASTIC DROP (FORWARD)
# ------------------------------------------------------------

def elastic_forward(Wo, par_ref, var_ref, mod_def, A_def, for_tol=1e-5, N_it=3, N_div=2, f_subdiv=5, A_exp=2, verbose=0):

    # Reference drop
    l_ref, P0_ref = par_ref
    psi_ref, r_ref, z_ref, v_ref = var_ref
    V = v_ref[-1]

    # Mesh
    N_mesh = len(psi_ref)
    eps = 1e-5
    s = np.linspace(eps, 1.0, N_mesh)

    # Reference drop area
    A_ref = np.trapezoid(2.0 * np.pi * r_ref * l_ref, s)

    # Deformation moduli
    K, G = mod_def

    # Reference profile interpolation
    r_ref_interp = sc.interpolate.interp1d(s, r_ref, kind="cubic", fill_value="extrapolate")
    psi_ref_interp = sc.interpolate.interp1d(s, psi_ref, kind="cubic", fill_value="extrapolate")

    # For returning special cases
    nonphysical_sol = {}
    last_converged_sol = {}

    # Differential equations
    def ODEs(s_interp, y, par):
        P0_def, lambda0 = par
        psi_def, z_def, lambda_xi, lambda_phi, A_def = y

        # Stretch ratios must remain positive
        if np.any(lambda_xi <= 0) or np.any(lambda_phi <= 0):
            raise RuntimeError("Negative stretch ratios")

        # Reference state
        r0 = r_ref_interp(s_interp) + 1e-12
        psi0 = psi_ref_interp(s_interp)

        # Relative area deformation
        J = lambda_xi * lambda_phi

        # Principal surface stresses
        Sxi = 1.0 + K * np.log(J) / J + 0.5 * G * (1.0 / lambda_phi**2 - 1.0 / lambda_xi**2)
        Sphi = 1.0 + K * np.log(J) / J - 0.5 * G * (1.0 / lambda_phi**2 - 1.0 / lambda_xi**2)

        # Stress derivatives
        dSxi_dlambda_xi = K * (1.0 - np.log(J)) / (lambda_xi**2 * lambda_phi) + G / lambda_xi**3
        dSxi_dlambda_phi = K * (1.0 - np.log(J)) / (lambda_xi * lambda_phi**2) - G / lambda_phi**3

        # Function derivatives
        dpsi_ds = lambda_xi * l_ref / Sxi * (P0_def - (2.0 * np.pi * Wo / V) * z_def - (Sphi * np.sin(psi_def)) / (r0 * lambda_phi))
        dz_ds = lambda_xi * l_ref * np.sin(psi_def)
        dlambda_xi_ds = (l_ref / (dSxi_dlambda_xi * r0 * lambda_phi)) * (lambda_xi * np.cos(psi_def) * (Sphi - Sxi) - dSxi_dlambda_phi * lambda_phi * (lambda_xi * np.cos(psi_def) - lambda_phi * np.cos(psi0)))
        dlambda_phi_ds = (l_ref / r0) * (lambda_xi * np.cos(psi_def) - lambda_phi * np.cos(psi0))

        # Accumulated interfacial area
        dA_ds = 2.0 * np.pi * r0 * lambda_phi * lambda_xi * l_ref

        return np.vstack((dpsi_ds, dz_ds, dlambda_xi_ds, dlambda_phi_ds, dA_ds))

    # Iterative continuation in area
    def forward_iterative(N_it, N_div, f_subdiv):
        it = 0

        while it < N_it:

            # Area mesh
            A_def_array = A_def / N_div + (A_def - A_def / N_div) * (1.0 - (1.0 - np.linspace(0.0, 1.0, N_div))**A_exp)

            # Initial guesses
            psi_init = psi_ref_interp(s)
            A_init = A_ref * np.sqrt(v_ref / v_ref[-1])
            lambda_xi_init = np.ones_like(s)
            lambda_phi_init = np.ones_like(s)
            var_est = np.vstack((psi_init, z_ref, lambda_xi_init, lambda_phi_init, A_init))
            par_est = np.array([P0_ref, 1.0], dtype=float)

            if verbose > 0:
                print(f"\nIteration {it + 1}/{N_it}")

            convergence = True

            for i, A_def_it in enumerate(A_def_array):

                if verbose > 1:
                    print(f"  [{i + 1}/{len(A_def_array)}] A_def={A_def_it}")

                A_target = A_ref * (1.0 - A_def_it)

                def BCs(ya, yb, par):
                    P0_def, lambda0 = par
                    
                    res = np.zeros(7)
                    #res[0] = ya[0] - (s[0] * l_ref * P0_def) / 2.0  # psi(eps)
                    sigma0 = 1 + K * np.log(lambda0**2) / lambda0**2
                    res[0] = ya[0] - (s[0] * l_ref * P0_def * lambda0) / (2.0 * sigma0) # psi(eps)
                    #res[1] = ya[1] - (s[0]**2 * l_ref**2 * P0_def) / 4.0  # z(eps)
                    res[1] = ya[1] - (s[0]**2 * l_ref**2 * P0_def * lambda0**2) / (4.0 * sigma0)  # z(eps)
                    res[2] = ya[2] - lambda0  # lambda_xi(eps)
                    res[3] = ya[3] - lambda0  # lambda_phi(eps)
                    res[4] = yb[3] - 1.0  # lambda_phi(1)
                    res[5] = ya[4] - np.pi * l_ref**2 * lambda0**2 * s[0]**2  # A(eps)
                    res[6] = yb[4] - A_target  # A(1)

                    return res

                try:
                    sol = sc.integrate.solve_bvp(fun=ODEs, bc=BCs, x=s, y=var_est, p=par_est, tol=for_tol, max_nodes=10000)

                    if sol.success:
                        var_est = sol.sol(s)
                        par_est = sol.p

                        # Store last converged solution
                        r_last = var_est[3] * r_ref_interp(s)
                        z_last = var_est[1]

                        last_converged_sol["par_def"] = np.array([par_est[0], par_est[1]])
                        last_converged_sol["var_def"] = np.array([var_est[0], r_last, z_last])
                        last_converged_sol["lambda_def"] = np.array([var_est[2], var_est[3]])
                        last_converged_sol["A_def"] = A_def_it

                        # Nonphysical solution
                        if np.argmax(var_est[1]) != len(var_est[1]) - 1:
                            nonphysical_sol["par_def"] = np.array([par_est[0], par_est[1]])
                            nonphysical_sol["var_def"] = np.array([var_est[0], r_last, z_last])
                            nonphysical_sol["lambda_def"] = np.array([var_est[2], var_est[3]])
                            nonphysical_sol["A_def"] = A_def_it

                            if verbose > 0:
                                print(f"\nNonphysical solution at A_def = {A_def_it}")

                            raise ValueError("nonphysical solution")

                    else:
                        convergence = False

                        if verbose > 0:
                            print(f"\nFailed to converge at A_def = {A_def_it}")
                            print(f"Message: {sol.message}")

                        break

                except RuntimeError as e:
                    convergence = False

                    if verbose > 0:
                        print(f"\nFailed to converge at A_def = {A_def_it}")
                        print(f"Message: {str(e)}")

                    break

            if convergence:
                if verbose > 0:
                    print("\nBVP converged")
                return par_est, var_est

            N_div *= f_subdiv
            it += 1

        raise RuntimeError("nonconvergent BVP")

    # Run solver
    try:
        par_best, var_best = forward_iterative(N_it=N_it, N_div=N_div, f_subdiv=f_subdiv)

        P0_def, lambda0 = par_best
        psi_def, z_def, lambda_xi, lambda_phi, _ = var_best

        r_def = lambda_phi * r_ref_interp(s)

        return {"message": "success", "par_def": np.array([P0_def, lambda0]), "var_def": np.array([psi_def, r_def, z_def]), "lambda_def": np.array([lambda_xi, lambda_phi])}

    except RuntimeError:
        print("\nERROR: nonconvergent BVP")

        if last_converged_sol:
            return {"message": "nonconvergent", "par_def": last_converged_sol["par_def"], "var_def": last_converged_sol["var_def"], "lambda_def": last_converged_sol["lambda_def"], "A_def_last": last_converged_sol["A_def"]}

        return {"message": "nonconvergent", "par_def": np.array([np.nan, np.nan]), "var_def": np.full((3, len(s)), np.nan), "lambda_def": np.full((2, len(s)), np.nan)}

    except ValueError:
        print("\nERROR: nonphysical solution")

        return {"message": "nonphysical", "par_def": nonphysical_sol["par_def"], "var_def": nonphysical_sol["var_def"], "lambda_def": nonphysical_sol["lambda_def"], "A_def_last": nonphysical_sol["A_def"]}


# ------------------------------------------------------------
# ELASTIC DROP - STRETCH (INVERSE)
# ------------------------------------------------------------

def elastic_inverse_stretch(Wo, V, var_def, for_tol=1e-5, inv_tol=1e-8, verbose=0):

    reference_drop = reference_forward(Wo=Wo, V=V)
    l_ref, P0_ref = reference_drop['par_ref']
    psi_ref, r_ref, z_ref, v_ref = reference_drop['var_ref']
    V_ref = np.copy(V)

    _, r_def, z_def = var_def

    # Mesh
    N_mesh = len(r_ref)
    eps = 1e-5
    s = np.linspace(eps, 1.0, N_mesh)

    penalty = 1e6 * np.ones(2 * N_mesh)

    # Reference profile interpolation
    r_ref_interp = sc.interpolate.interp1d(s, r_ref, kind="cubic", fill_value="extrapolate")
    psi_ref_interp = sc.interpolate.interp1d(s, psi_ref, kind="cubic", fill_value="extrapolate")

    # Translated target profile only for comparison
    z_def_cmp = z_def - z_def[-1]

    def solve_forward(K, G, lambda0_target):

        def ODEs(s_interp, y, par):
            P0 = par[0]
            psi, z, lambda_xi, lambda_phi = y

            if np.any(lambda_xi <= 0) or np.any(lambda_phi <= 0):
                raise ValueError("nonpositive stretch")

            r0 = r_ref_interp(s_interp) + 1e-12
            psi0 = psi_ref_interp(s_interp)

            J = lambda_xi * lambda_phi

            if np.any(J <= 0):
                raise ValueError("nonpositive J")

            Sxi = 1.0 + K * np.log(J) / J + 0.5 * G * (1.0 / lambda_phi**2 - 1.0 / lambda_xi**2)
            Sphi = 1.0 + K * np.log(J) / J - 0.5 * G * (1.0 / lambda_phi**2 - 1.0 / lambda_xi**2)

            dSxi_dlambda_xi = K * (1.0 - np.log(J)) / (lambda_xi**2 * lambda_phi) + G / lambda_xi**3
            dSxi_dlambda_phi = K * (1.0 - np.log(J)) / (lambda_xi * lambda_phi**2) - G / lambda_phi**3

            dpsi_ds = (lambda_xi * l_ref / Sxi) * (P0 - (2.0 * np.pi * Wo / V_ref) * z - (Sphi * np.sin(psi)) / (r0 * lambda_phi))
            dz_ds = lambda_xi * l_ref * np.sin(psi)
            dlambda_xi_ds = (l_ref / (dSxi_dlambda_xi * r0 * lambda_phi)) * (lambda_xi * np.cos(psi) * (Sphi - Sxi) - dSxi_dlambda_phi * lambda_phi * (lambda_xi * np.cos(psi) - lambda_phi * np.cos(psi0)))
            dlambda_phi_ds = (l_ref / r0) * (lambda_xi * np.cos(psi) - lambda_phi * np.cos(psi0))

            return np.vstack((dpsi_ds, dz_ds, dlambda_xi_ds, dlambda_phi_ds))

        def BCs(ya, yb, par):
            P0 = par[0]

            res = np.zeros(5)
            sigma0 = 1 + K * np.log(lambda0_target**2) / lambda0_target**2
            res[0] = ya[0] - (s[0] * l_ref * P0 * lambda0_target) / (2.0 * sigma0) # psi(eps)
            res[1] = ya[1] - (s[0]**2 * l_ref**2 * P0 * lambda0_target**2) / (4.0 * sigma0)  # z(eps)
            res[2] = ya[2] - lambda0_target  # lambda_xi(eps)
            res[3] = ya[3] - lambda0_target  # lambda_phi(eps)
            res[4] = yb[3] - 1.0  # lambda_phi(1)

            return res

        # Initial guesses
        psi_est = psi_ref_interp(s)
        z_est = z_ref.copy()
        lambda_xi_est = np.ones_like(s)
        lambda_phi_est = np.ones_like(s)

        y_est = np.vstack((psi_est, z_est, lambda_xi_est, lambda_phi_est))
        par_init = np.array([P0_ref], dtype=float)

        sol = sc.integrate.solve_bvp(ODEs, BCs, s, y_est, p=par_init, tol=for_tol, max_nodes=10000, verbose=0)

        P0_comp = sol.p[0]
        psi, z, lambda_xi, lambda_phi = sol.sol(s)

        r = lambda_phi * r_ref

        return {"success": bool(sol.success), "message": sol.message, "P0": P0_comp, "var_comp": np.array([psi, r, z]), "lambda_comp": np.array([lambda_xi, lambda_phi])}

    def residuals_logpar(x):
        logK, logG, lambda0 = x

        if not np.all(np.isfinite(x)):
            return penalty

        K = 10.0**logK
        G = 10.0**logG

        if (K <= 0) or (G <= 0) or (lambda0 <= 0):
            return penalty

        try:
            forward_result = solve_forward(K, G, lambda0)
            _, r_calc, z_calc = forward_result["var_comp"]
        except Exception:
            return penalty
        
        z_calc_cmp = z_calc - z_calc[-1]

        scale_r = np.max(np.abs(r_def))
        scale_z = np.max(np.abs(z_def_cmp))

        res_r = (r_calc - r_def) / scale_r
        res_z = (z_calc_cmp - z_def_cmp) / scale_z

        res = np.concatenate((res_r, res_z))
        
        return res

    # Initial estimate for lambda0
    n_ap = 10
    lambda0_est = np.trapezoid(r_def[:n_ap], s[:n_ap]) / np.trapezoid(r_ref[:n_ap], s[:n_ap])
    
    # Initial optimization guess
    K0, G0 = 0.1, 0.1
    x0 = np.array([np.log10(K0), np.log10(G0), lambda0_est], dtype=float)

    lb = np.array([np.log10(0.01), np.log10(0.01), 0.5 * lambda0_est], dtype=float)
    ub = np.array([np.log10(3.), np.log10(3.), 1.5 * lambda0_est], dtype=float)

    ajuste = sc.optimize.least_squares(residuals_logpar, x0=x0, bounds=(lb, ub), ftol=inv_tol, xtol=inv_tol, gtol=inv_tol, max_nfev=100, verbose=verbose)

    logK_aj, logG_aj, lambda0_aj = ajuste.x

    K_aj = 10.0**logK_aj
    G_aj = 10.0**logG_aj

    if verbose > 0:
        print("Ajuste final:")
        print(f"K = {K_aj:.6f}")
        print(f"G = {G_aj:.6f}")
        print(f"lambda0 = {lambda0_aj:.6f}")
        print(f"least_squares success = {ajuste.success}")
        print(f"nfev = {ajuste.nfev}")
        
    return {"success": bool(ajuste.success), "message": f"nfev = {ajuste.nfev}, status = {ajuste.status}", "fit": np.array([K_aj, G_aj, lambda0_aj], dtype=float)}
    
# ------------------------------------------------------------
# ELASTIC DROP - AREA (INVERSE)
# ------------------------------------------------------------

def elastic_inverse_area(Wo, V, var_def, for_tol=1e-5, inv_tol=1e-8, verbose=0):

    # Reference drop
    reference_drop = reference_forward(Wo=Wo, V=V)
    l_ref, P0_ref = reference_drop['par_ref']
    psi_ref, r_ref, z_ref, v_ref = reference_drop['var_ref']

    _, r_def, z_def = var_def

    # Mesh
    N_mesh = len(r_ref)
    eps = 1e-5
    s = np.linspace(eps, 1.0, N_mesh)

    penalty = 1e6 * np.ones(2 * N_mesh)

    # Reference drop area
    A_ref = np.trapezoid(
        2.0 * np.pi * r_ref * l_ref,
        s
    )

    # Target deformed area inferred from target profile
    dr_def_ds = np.gradient(r_def, s)
    dz_def_ds = np.gradient(z_def, s)
    dl_def_ds = np.sqrt(dr_def_ds**2 + dz_def_ds**2)

    A_target = np.trapezoid(
        2.0 * np.pi * r_def * dl_def_ds,
        s
    )

    A_def = 1.0 - A_target / A_ref

    if verbose > 0:
        print("Área de referencia:", A_ref)
        print("Área objetivo:", A_target)
        print("A_def inferido:", A_def)

    # Translated target profile only for comparison
    z_def_cmp = z_def - z_def[-1]

    def residuals_logpar(x):

        logK, logG = x

        if not np.all(np.isfinite(x)):
            return penalty

        K = 10.0**logK
        G = 10.0**logG

        if (K <= 0) or (G <= 0):
            return penalty

        try:

            forward_result = elastic_forward(
                Wo=Wo,
                par_ref=reference_drop['par_ref'],
                var_ref=reference_drop['var_ref'],
                mod_def=np.array([K, G]),
                A_def=A_def,
                for_tol=for_tol,
                verbose=0
            )

            if forward_result["message"] != "success":
                return penalty

            _, r_calc, z_calc = forward_result["var_def"]

        except Exception:

            return penalty

        z_calc_cmp = z_calc - z_calc[-1]

        scale_r = np.max(np.abs(r_def))
        scale_z = np.max(np.abs(z_def_cmp))

        if scale_r == 0 or scale_z == 0:
            return penalty

        res_r = (r_calc - r_def) / scale_r
        res_z = (z_calc_cmp - z_def_cmp) / scale_z

        res = np.concatenate((res_r, res_z))

        if not np.all(np.isfinite(res)):
            return penalty

        return res

    # Initial optimization guess
    K0, G0 = 0.1, 0.1
    x0 = np.array(
        [
            np.log10(K0),
            np.log10(G0)
        ],
        dtype=float
    )

    # Bounds
    lb = np.array(
        [
            np.log10(0.01),
            np.log10(0.01)
        ],
        dtype=float
    )

    ub = np.array(
        [
            np.log10(3.0),
            np.log10(3.0)
        ],
        dtype=float
    )

    ajuste = sc.optimize.least_squares(
        residuals_logpar,
        x0=x0,
        bounds=(lb, ub),
        ftol=inv_tol,
        xtol=inv_tol,
        gtol=inv_tol,
        max_nfev=100,
        verbose=verbose
    )

    logK_aj, logG_aj = ajuste.x

    K_aj = 10.0**logK_aj
    G_aj = 10.0**logG_aj

    # Compute lambda0 from the best forward solution
    try:

        forward_best = elastic_forward(
            Wo=Wo,
            par_ref=reference_drop['par_ref'],
            var_ref=reference_drop['var_ref'],
            mod_def=np.array([K_aj, G_aj]),
            A_def=A_def,
            for_tol=for_tol,
            verbose=0
        )

        if forward_best["message"] == "success":

            P0_aj, lambda0_aj = forward_best["par_def"]

        else:

            P0_aj = np.nan
            lambda0_aj = np.nan

    except Exception:

        forward_best = {}
        P0_aj = np.nan
        lambda0_aj = np.nan

    if verbose > 0:
        print("Ajuste final:")
        print(f"K = {K_aj:.6f}")
        print(f"G = {G_aj:.6f}")
        print(f"lambda0 = {lambda0_aj:.6f}")
        print(f"A_def = {A_def:.6f}")
        print(f"least_squares success = {ajuste.success}")
        print(f"nfev = {ajuste.nfev}")

    return {
        "success": bool(ajuste.success),
        "message": f"nfev = {ajuste.nfev}, status = {ajuste.status}",
        "fit": np.array([K_aj, G_aj, lambda0_aj], dtype=float),
        "A_ref": A_ref,
        "A_target": A_target,
        "A_def": A_def,
        "par_def": np.array([P0_aj, lambda0_aj], dtype=float),
        "forward_best": forward_best
    }
