"""Run the reproduction and produce the figures for the manuscript.

Figures
-------
figures/fig1_hysteresis.png   : mean-field fixed-point branches vs. lambda (S-curve)
                                with agent-based forward/backward hysteresis sweeps.
figures/fig2_phase_diagram.png: bistability region in the (lambda, lambda_D) plane.
figures/fig3_time_series.png  : time evolution from different initial seeds
                                (critical-mass / memory effect).
figures/network_stats.json    : measured network statistics and key numbers.
"""

import json
import os

import numpy as np

import simplicial_contagion as sc

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters (following Iacopini et al. 2019)
# ---------------------------------------------------------------------------
N = 2000
KBAR = 10          # mean pairwise degree  <k>
KDBAR = 10         # mean number of incident 2-simplices <k_D>
LAMD = 0.6         # 2-simplex infection rate for the S-curve
SEED = 42
T_MAX = 400.0
DT = 0.02

rng = np.random.default_rng(SEED)
adj, tris = sc.generate_simplicial_complex(N, KBAR, KDBAR, rng)

# measured statistics
kbar_meas = adj.sum(axis=1).mean()
kdbar_meas = np.bincount(tris.reshape(-1), minlength=N).mean()
m_tris = tris.shape[0]
print(f"[net] N={N}  M_links={adj.nnz//2}  M_2simplex={m_tris}")
print(f"[net] <k>={kbar_meas:.2f} (target {KBAR})   <k_D>={kdbar_meas:.2f} (target {KDBAR})")

# ---------------------------------------------------------------------------
# Fig 1 -- mean-field S-curve + hysteresis sweeps
# ---------------------------------------------------------------------------
fig1_path = os.path.join(OUT, "fig1_hysteresis.png")
if os.path.exists(fig1_path):
    print("[fig1] already exists, skipping hysteresis sweeps")
    fwd = back = lam_asc = None
else:
    lam_vals = np.linspace(0.0, 0.15, 21)
    lam_asc, fwd, back = sc.sweep_hysteresis(adj, tris, KBAR, KDBAR, lam_vals, LAMD,
                                             t_max=T_MAX, dt=DT)

stable_branches, unstable_branches = [], []
lam_th = np.linspace(0.0, 0.16, 400)
for lam in lam_th:
    st, un = sc.mf_fixed_points(lam, LAMD, KBAR, KDBAR)
    stable_branches.append(st)
    unstable_branches.append(un)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
})

fig, ax = plt.subplots(figsize=(6.4, 4.6))
# stable branches (up to two positive + possibly rho=0)
for i in range(3):
    ys = [b[i] if i < len(b) else np.nan for b in stable_branches]
    ax.plot(lam_th, ys, color="#1f77b4", lw=2.0, label="stable" if i == 0 else None)
for i in range(2):
    ys = [u[i] if i < len(u) else np.nan for u in unstable_branches]
    ax.plot(lam_th, ys, "--", color="#d62728", lw=1.8, label="unstable" if i == 0 else None)

if fwd is not None:
    ax.scatter(lam_asc, fwd, marker="o", s=46, facecolors="none", edgecolors="#2ca02c",
               linewidths=1.4, zorder=5, label="simulation: forward sweep")
    ax.scatter(lam_asc, back, marker="s", s=40, facecolors="none", edgecolors="#ff7f0e",
               linewidths=1.4, zorder=5, label="simulation: backward sweep")

ax.set_xlabel(r"pairwise infection rate  $\lambda$")
ax.set_ylabel(r"steady-state infected fraction  $\rho_\infty$")
ax.set_title(rf"Higher-order contagion: $\lambda_\Delta={LAMD}$, "
             rf"$\langle k\rangle={KBAR}$, $\langle k_\Delta\rangle={KDBAR}$")
ax.set_xlim(0, 0.155)
ax.set_ylim(-0.03, 1.03)
ax.legend(loc="center right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_hysteresis.png"), dpi=200)
plt.close(fig)
print("[fig1] saved")

# ---------------------------------------------------------------------------
# Fig 2 -- phase diagram (bistability region, mean field)
# ---------------------------------------------------------------------------
# Vectorized analytic criterion (mean field):
#   f(rho) = rho * g(rho),  g(rho) = A*rho^2 + B*rho + C
#   A = -lamd*KDBAR,  B = lamd*KDBAR - lam*KBAR,  C = lam*KBAR - 1
# rho=0 stable  <=>  C < 0 ; two interior roots of g with stability of the
# upper branch  <=>  disc > 0 and 0 < r_small < r_large < 1.
lam_grid = np.linspace(0.0, 0.15, 500)
lamd_grid = np.linspace(0.0, 1.6, 500)
LAM, LAMD2 = np.meshgrid(lam_grid, lamd_grid, indexing="ij")
A = -LAMD2 * KDBAR
B = LAMD2 * KDBAR - LAM * KBAR
C = LAM * KBAR - 1.0
disc = B * B - 4.0 * A * C
with np.errstate(divide="ignore", invalid="ignore"):
    sqrtd = np.sqrt(np.maximum(disc, 0.0))
    r_large = (-B - sqrtd) / (2.0 * A)
    r_small = (-B + sqrtd) / (2.0 * A)
bistable = (A < 0) & (C < 0) & (disc > 0) & (r_small > 0) & (r_large < 1.0)
bistable = bistable.T  # shape (len(lamd_grid), len(lam_grid)) for pcolormesh

fig, ax = plt.subplots(figsize=(6.4, 4.6))
im = ax.pcolormesh(lam_grid, lamd_grid, bistable, cmap="Blues", shading="auto", vmin=0, vmax=1)
cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
cbar.ax.set_yticklabels(["monostable", "bistable"])
# pairwise epidemic threshold line lambda_c = 1/<k>
ax.axvline(1.0 / KBAR, color="#d62728", ls="--", lw=1.5,
           label=r"pairwise threshold $\lambda_c=1/\langle k\rangle$")
# analytic saddle-node line:  (y-x)^2 + 4y(x-1) = 0 with x=lam<k>, y=lamD<kD>
# upper branch: y = 2 - x + 2*sqrt(1-x),  x in [0, 1]
x_sn = np.linspace(0.0, 1.0 - 1e-9, 400)
y_sn = 2.0 - x_sn + 2.0 * np.sqrt(1.0 - x_sn)
ax.plot(x_sn / KBAR, y_sn / KDBAR, color="#1f77b4", lw=1.8,
        label="saddle-node line")
ax.set_xlabel(r"pairwise infection rate  $\lambda$")
ax.set_ylabel(r"2-simplex infection rate  $\lambda_\Delta$")
ax.set_title(rf"Bistability region in the ($\lambda,\lambda_\Delta$) plane, "
             rf"$\langle k\rangle={KBAR}$, $\langle k_\Delta\rangle={KDBAR}$")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_phase_diagram.png"), dpi=200)
plt.close(fig)
print("[fig2] saved")

# ---------------------------------------------------------------------------
# Fig 3 -- critical mass: time series from different initial seeds
# ---------------------------------------------------------------------------
LAM_FIX = 0.04
rhos0 = [0.05, 0.08, 0.10, 0.12, 0.14, 0.20, 0.30]  # matches Table 1 seed set
fig, ax = plt.subplots(figsize=(6.4, 4.6))
for i, r0 in enumerate(rhos0):
    ts, rhos = sc.simulate(adj, tris, LAM_FIX, LAMD, r0, t_max=300.0, dt=DT)
    ax.plot(ts, rhos, lw=1.6, label=rf"$\rho_0={r0}$")
ax.set_xlabel(r"time  $t$")
ax.set_ylabel(r"infected fraction  $\rho(t)$")
ax.set_title(rf"Critical-mass effect: $\lambda={LAM_FIX}$, "
             rf"$\lambda_\Delta={LAMD}$")
ax.legend(loc="right")
ax.set_ylim(-0.02, 1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_time_series.png"), dpi=200)
plt.close(fig)
print("[fig3] saved")

# ---------------------------------------------------------------------------
# Key numbers for the manuscript
# ---------------------------------------------------------------------------
lam_c_pair = 1.0 / KBAR
# bistability window at lambda_D = 0.6 (mean field): vectorized scan
lam_scan = np.linspace(0.0, 0.15, 1501)
A_w = -LAMD * KDBAR * np.ones_like(lam_scan)
B_w = LAMD * KDBAR - lam_scan * KBAR
C_w = lam_scan * KBAR - 1.0
disc_w = B_w * B_w - 4.0 * A_w * C_w
sqrt_w = np.sqrt(np.maximum(disc_w, 0.0))
rL_w = (-B_w - sqrt_w) / (2.0 * A_w)
rS_w = (-B_w + sqrt_w) / (2.0 * A_w)
mask_w = (C_w < 0) & (disc_w > 0) & (rS_w > 0) & (rL_w < 1.0)
window = (float(lam_scan[mask_w].min()), float(lam_scan[mask_w].max())) if mask_w.any() else (None, None)

stats = {
    "N": N,
    "kbar_target": KBAR,
    "kbar_measured": float(kbar_meas),
    "kdbar_target": KDBAR,
    "kdbar_measured": float(kdbar_meas),
    "M_2simplex": int(m_tris),
    "lambda_D": LAMD,
    "lambda_c_pairwise": float(lam_c_pair),
    "bistable_lambda_window_at_lamD_0.6": [window[0], window[1]],
    "hysteresis_forward_final_rho": None if fwd is None else float(fwd[-1]),
    "hysteresis_backward_final_rho": None if back is None else float(back[-1]),
}
with open(os.path.join(OUT, "network_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)
print("[stats] saved:", json.dumps(stats, indent=2))
print("ALL DONE")
