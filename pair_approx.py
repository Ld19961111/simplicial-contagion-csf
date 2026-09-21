# -*- coding: utf-8 -*-
"""Direction 1 -- pair approximation of the critical mass in simplicial contagion.

The homogeneous mean-field (0.1235) and the network-resolved rate equations
(0.110-0.113) both assume effectively independent neighbours.  Real dynamics
build pairwise correlations (infection clustering), which changes how a
susceptible node "sees" infected neighbours.  We write a pair approximation
that tracks:

    rho = [I]                       infected density
    SI  = [SI]                      fraction of (undirected) edges that are S-I
    II  = [II]                      fraction of edges that are I-I
    T1, T2, T3                      fractions of 2-simplices with 1/2/3 infected

closed with:
  * edge triplets: <S I X> ~ k*[SI]*[IX]/[I]  (Kirkwood on edges), expressed
    through conditional infection probability p_SI = [SI]/[S] for the *other*
    neighbours of an S node,
  * triangle transitions exact inside the triangle; the external field of a
    triangle node is taken from the global edge/triangle densities.

Regular-network approximation: k = <k>, kD = <kD> (both 10 here).

Output:
  figures/pa_stats.json      -- rho*_PA vs rho*_ABM/NRMF/MF for the baseline
                                and (preliminary) structural variations
  figures/fig10_pairapprox.png
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

KBAR = 10.0
KDBAR = 10.0
LAM = 0.04
LAMD = 0.6
T_MAX = 300.0
DT_ODE = 0.005


def pair_approx_rhs(t, y, k, kd, lam, lamd):
    """y = [rho, SI, II, T1, T2, T3]"""
    rho, SI, II, T1, T2, T3 = y
    S = 1.0 - rho
    SS = 1.0 - 2.0 * SI - II
    T0 = 1.0 - T1 - T2 - T3
    pSI = SI / S if S > 1e-12 else 0.0
    c2 = (kd / 3.0) * T2            # mean number of 2-infected triangles per node

    # susceptible-node activation rates in the different local contexts
    r_SS = lam * ((k - 1.0) * pSI) + lamd * c2            # SS edge, either endpoint
    r_SI = lam * (1.0 + (k - 1.0) * pSI) + lamd * c2      # SI edge, S endpoint
    r_T0 = lam * (k * pSI) + lamd * c2                    # S in a T0 triangle
    r_T1 = lam * (1.0 + (k - 2.0) * pSI) + lamd * c2
    r_T2 = lam * (2.0 + (k - 2.0) * pSI) + lamd * (1.0 + c2)

    drho = -rho + lam * k * SI + lamd * c2
    dSI = SS * r_SS + II * 1.0 - SI * (r_SI + 1.0)
    dII = SI * r_SI - 2.0 * II
    dT1 = 3.0 * T0 * r_T0 + T2 * 1.0 - T1 * (r_T1 + 1.0)
    dT2 = T1 * r_T1 + 3.0 * T3 - T2 * (r_T2 + 1.0)
    dT3 = T2 * r_T2 - 3.0 * T3
    return np.array([drho, dSI, dII, dT1, dT2, dT3])


def init_from_seed(rho0):
    S = 1.0 - rho0
    SI = rho0 * S
    II = rho0 * rho0
    T1 = 3.0 * rho0 * S * S
    T2 = 3.0 * rho0 * rho0 * S
    T3 = rho0 ** 3
    return np.array([rho0, SI, II, T1, T2, T3])


def integrate(rho0, k=KBAR, kd=KDBAR, lam=LAM, lamd=LAMD, t_max=T_MAX):
    y = init_from_seed(rho0)
    n_steps = int(t_max / DT_ODE)
    for _ in range(n_steps):
        y = y + DT_ODE * pair_approx_rhs(0.0, y, k, kd, lam, lamd)
        y = np.clip(y, 0.0, 1.0)
    return y[0]


def pa_critical(k=KBAR, kd=KDBAR, lam=LAM, lamd=LAMD, lo=0.02, hi=0.35, levels=40):
    """Bisect the pair-approximation critical mass (survival: final rho > 0.5)."""
    for _ in range(levels):
        mid = 0.5 * (lo + hi)
        if integrate(mid, k, kd, lam, lamd) > 0.5:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def mf_critical(lam=LAM, lamd=LAMD, k=KBAR, kd=KDBAR):
    """Mean-field unstable fixed point (rho*_MF): solve
    lamd*kd*rho^2 + (lam*k - lamd*kd)*rho + (1 - lam*k) = 0."""
    a = lamd * kd
    b = lam * k - lamd * kd
    c = 1.0 - lam * k
    disc = b * b - 4.0 * a * c
    if disc <= 0:
        return None
    r1 = (-b + np.sqrt(disc)) / (2.0 * a)
    r2 = (-b - np.sqrt(disc)) / (2.0 * a)
    return min(r1, r2)


def main():
    rho_pa = pa_critical()
    rho_mf = mf_critical()
    print(f"[pa] rho*_PA  = {rho_pa:.4f}")
    print(f"[pa] rho*_MF  = {rho_mf:.4f}")

    # validate the trajectory shape at a few seeds
    for r0 in (0.05, 0.12, 0.15, 0.18, 0.25):
        rf = integrate(r0)
        print(f"[pa] seed {r0:.2f} -> final {rf:.3f}")

    # structural sensitivity: effect of hyperdegree variance (crude proxy:
    # effective kd in the triangle closure) and overlap (more edge-shared
    # triangles -> triangle state coupled to edges; approximated by increasing
    # the effective kd of the edge channel lambda term)
    variants = {}
    for kd_eff in (8.0, 10.0, 12.0):
        rp = pa_critical(kd=kd_eff)
        variants[f"kd_eff_{kd_eff:.0f}"] = rp
        print(f"[pa] kd_eff={kd_eff}: rho*_PA={rp:.4f}")

    with open(os.path.join(OUT, "pa_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"params": {"k": KBAR, "kd": KDBAR, "lam": LAM, "lamd": LAMD,
                              "t_max": T_MAX, "dt": DT_ODE},
                   "rho_star_PA": float(rho_pa),
                   "rho_star_MF": float(rho_mf),
                   "variants_kd": variants}, f, indent=2)
    print("[pa] saved pa_stats.json")

    # ---- figure: pair-approx S-curve (low/high seed) vs mean field ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
                         "legend.fontsize": 9.5, "mathtext.fontset": "stix",
                         "font.family": "STIXGeneral"})
    lam_grid = np.linspace(0.005, 0.16, 40)
    pa_lo = [integrate(0.005, lam=lm) for lm in lam_grid]
    pa_hi = [integrate(0.95, lam=lm) for lm in lam_grid]

    def mf_rho(lam, rho0):
        # iterate the mean-field map to fixed point
        r = rho0
        for _ in range(20000):
            f = -r + lam * KBAR * r * (1 - r) + LAMD * KDBAR * r * r * (1 - r)
            r = np.clip(r + 0.005 * f, 0, 1)
        return r

    mf_lo = [mf_rho(lm, 0.005) for lm in lam_grid]
    mf_hi = [mf_rho(lm, 0.95) for lm in lam_grid]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(lam_grid, mf_lo, color="#7f7f7f", lw=2.0, label="mean field: low seed")
    ax.plot(lam_grid, mf_hi, color="#7f7f7f", ls="--", lw=2.0, label="mean field: high seed")
    ax.plot(lam_grid, pa_lo, color="#1f77b4", lw=2.0, label="pair approx: low seed")
    ax.plot(lam_grid, pa_hi, color="#1f77b4", ls="--", lw=2.0, label="pair approx: high seed")
    ax.axvline(0.1, color="#d62728", ls=":", lw=1.5, label=r"$\lambda_c=1/\langle k\rangle$")
    ax.set_xlabel(r"pairwise rate $\lambda$")
    ax.set_ylabel(r"final density $\rho_\infty$")
    ax.set_title(r"Pair approximation vs mean field, $\lambda_\Delta\langle k_\Delta\rangle=6$")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92, edgecolor="#cccccc")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig10_pairapprox.png"), dpi=200)
    plt.close(fig)
    print("[pa] fig10 saved")
    print("[pa] ALL DONE")


if __name__ == "__main__":
    main()
