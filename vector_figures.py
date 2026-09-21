# -*- coding: utf-8 -*-
"""Re-render all five combined figures as VECTOR graphics (PDF + EPS).

Replaces the PIL-pasted PNG composites. Every panel is drawn natively in
matplotlib inside a single subplots figure, so fonts/lines/markers are real
vector primitives. Data come from the same *.json files; the few panels that
require a quick simulation (hysteresis sweep, time series, cascade hysteresis,
PA S-curve) are recomputed with the same seeds/protocol.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simplicial_contagion as sc
import cascade_infrastructure as ci
import pair_approx as pa

OUT = os.path.join(HERE, "figures")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 9, "mathtext.fontset": "stix", "font.family": "STIXGeneral",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save_vector(fig, stem):
    """Save as both .pdf (vector, for xelatex) and .eps (for submission)."""
    for ext in (".pdf", ".eps"):
        fig.savefig(os.path.join(OUT, stem + ext), bbox_inches="tight")
    plt.close(fig)
    print("[vector]", stem + ".pdf/.eps")


def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as fh:
        return json.load(fh)


# ============================================================== Fig 1 =====
KBAR, KDBAR, LAMD = 10.0, 10.0, 0.6
N = 2000
rng = np.random.default_rng(42)
adj, tris = sc.generate_simplicial_complex(N, KBAR, KDBAR, rng)

lam_th = np.linspace(0.0, 0.16, 400)
stab, unst = [], []
for lam in lam_th:
    s, u = sc.mf_fixed_points(lam, LAMD, KBAR, KDBAR)
    stab.append(s); unst.append(u)

lam_asc = np.linspace(0.0, 0.15, 21)
lam_asc, fwd, back = sc.sweep_hysteresis(adj, tris, KBAR, KDBAR, lam_asc, LAMD,
                                         t_max=400.0, dt=0.02)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
for i in range(3):
    ys = [b[i] if i < len(b) else np.nan for b in stab]
    ax1.plot(lam_th, ys, color="#1f77b4", lw=2.0, label="stable" if i == 0 else None)
for i in range(2):
    ys = [u[i] if i < len(u) else np.nan for u in unst]
    ax1.plot(lam_th, ys, "--", color="#d62728", lw=1.8,
             label="unstable" if i == 0 else None)
ax1.scatter(lam_asc, fwd, marker="o", s=40, facecolors="none",
            edgecolors="#2ca02c", linewidths=1.3, zorder=5,
            label="forward sweep")
ax1.scatter(lam_asc, back, marker="s", s=34, facecolors="none",
            edgecolors="#ff7f0e", linewidths=1.3, zorder=5,
            label="backward sweep")
ax1.set_xlabel(r"$\lambda$"); ax1.set_ylabel(r"$\rho_\infty$")
ax1.set_title(r"(a)", loc="left", fontweight="bold")
ax1.set_xlim(0, 0.155); ax1.set_ylim(-0.03, 1.03)
ax1.legend(loc="lower right", fontsize=8.5)

lg = np.linspace(0.0, 0.15, 500); ldg = np.linspace(0.0, 1.6, 500)
LAM, LD = np.meshgrid(lg, ldg, indexing="ij")
A = -LD * KDBAR; B = LD * KDBAR - LAM * KBAR; C = LAM * KBAR - 1.0
disc = B*B - 4*A*C
sq = np.sqrt(np.maximum(disc, 0.0))
rL = (-B - sq) / (2*A); rS = (-B + sq) / (2*A)
bist = (A < 0) & (C < 0) & (disc > 0) & (rS > 0) & (rL < 1.0)
ax2.pcolormesh(lg, ldg, bist.T, cmap="Blues", shading="auto", vmin=0, vmax=1)
ax2.axvline(1.0/KBAR, color="#d62728", ls="--", lw=1.5,
            label=r"$\lambda_c=1/\langle k\rangle$")
xs = np.linspace(0.0, 1.0-1e-9, 400)
ax2.plot(xs/KBAR, (2.0-xs+2*np.sqrt(1.0-xs))/KDBAR, color="#1f77b4", lw=1.8,
         label="saddle-node line")
ax2.set_xlabel(r"$\lambda$"); ax2.set_ylabel(r"$\lambda_\Delta$")
ax2.set_title(r"(b)", loc="left", fontweight="bold")
ax2.legend(loc="lower left", fontsize=8.5)
fig.tight_layout()
save_vector(fig, "fig_combined_1")

# ============================================================== Fig 2 =====
rhos0 = [0.05, 0.08, 0.10, 0.12, 0.14, 0.20, 0.30]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
for r0 in rhos0:
    ts, rhos = sc.simulate(adj, tris, 0.04, LAMD, r0, t_max=300.0, dt=0.02)
    ax1.plot(ts, rhos, lw=1.5, label=rf"$\rho_0={r0}$")
ax1.set_xlabel(r"time $t$"); ax1.set_ylabel(r"$\rho(t)$")
ax1.set_title(r"(a)", loc="left", fontweight="bold")
ax1.set_ylim(-0.02, 1.02); ax1.legend(loc="center right", fontsize=8)

FS = load("finite_size_stats.json")
for n, d in FS["per_N"].items():
    for v in d["rho_star_values"]:
        ax2.plot(float(n), v, "o", color="#1f77b4", ms=5, alpha=0.6)
Ns = np.array([float(x) for x in FS["per_N"]])
means = np.array([FS["per_N"][str(int(n))]["mean"] for n in Ns])
stds = np.array([FS["per_N"][str(int(n))]["std"] for n in Ns])
ax2.errorbar(Ns, means, yerr=stds, fmt="s", color="#d62728", ms=7,
             capsize=4, lw=1.5, label="agent-based")
nn = np.linspace(Ns.min(), Ns.max(), 200)
ax2.plot(nn, FS["fit"]["A"] + FS["fit"]["B"]*nn**(-FS["fit"]["alpha"]),
         "--", color="#2ca02c", lw=2.0,
         label=rf"fit $\rho^*_\infty$={FS['fit']['A']:.3f}")
ax2.axhline(0.1235, color="#7f7f7f", ls=":", lw=1.5,
            label=r"MF $\rho^*=0.124$")
ax2.set_xscale("log"); ax2.set_xlabel(r"system size $N$")
ax2.set_ylabel(r"$\rho^*(N)$")
ax2.set_title(r"(b)", loc="left", fontweight="bold")
ax2.set_ylim(0.11, 0.195)
ax2.legend(loc="lower left", fontsize=8)
fig.tight_layout()
save_vector(fig, "fig_combined_2")

# ============================================================== Fig 3 =====
g118 = __import__("networkx").Graph()
g118.add_edges_from(ci.ieee118_edges())
pos = ci.__dict__.get("_pos_cache") or None
import networkx as nx
pos = nx.spring_layout(g118, seed=42, k=0.9)

t118_adj, t118_tris = ci.build(g118, np.random.default_rng(2025))
kbar1 = t118_adj.sum(axis=1).mean(); kdbar1 = np.bincount(t118_tris.reshape(-1), minlength=118).mean()
lamd5 = 5.0/kdbar1
lam_v = np.linspace(0.0, 0.55, 23)
fwd = [ci.sweep_point(t118_adj, t118_tris, lamd5, la, 0.01) for la in lam_v]
bwd = [ci.sweep_point(t118_adj, t118_tris, lamd5, la, 0.95) for la in lam_v[::-1]][::-1]
lam_th = np.linspace(0.0, 0.55, 60)
nlo = [ci.nrmf_final(t118_adj, t118_tris, 0.005, la, lamd5) for la in lam_th]
nhi = [ci.nrmf_final(t118_adj, t118_tris, 0.9, la, lamd5) for la in lam_th]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
d = [g118.degree(u) for u in g118]
nx.draw_networkx_edges(g118, pos, ax=axes[0], edge_color="#b8c4d8", width=0.7)
nx.draw_networkx_nodes(g118, pos, ax=axes[0], node_size=55, node_color=d,
                       cmap="viridis", vmin=min(d), vmax=max(d))
axes[0].set_title(r"(a)", loc="left", fontweight="bold"); axes[0].axis("off")

axes[1].plot(lam_th, nlo, color="#1f77b4", lw=2.0, label="rate eq. low")
axes[1].plot(lam_th, nhi, color="#1f77b4", ls="--", lw=2.0, label="rate eq. high")
axes[1].plot(lam_v, fwd, "o", ms=5, color="#2ca02c", label="AB forward")
axes[1].plot(lam_v, bwd, "s", ms=5, color="#ff7f0e", label="AB backward")
axes[1].axvline(1.0/kbar1, color="#d62728", ls=":", lw=1.5,
                label=r"$\lambda_c=1/\langle k\rangle$")
axes[1].set_xlabel(r"$\lambda$"); axes[1].set_ylabel(r"$\rho_\infty$")
axes[1].set_title(r"(b)", loc="left", fontweight="bold"); axes[1].set_ylim(-0.03, 1.03)
axes[1].legend(loc="lower right", fontsize=7.5)

CS = load("cascade_stats.json")
colors = {"IEEE118": "#d62728", "ER(118)": "#1f77b4", "BA(118)": "#2ca02c"}
for name, r in CS["results"].items():
    axes[2].plot(CS["y_grid"], r["rho_star_AB"], "o-", color=colors[name],
                 ms=6, lw=1.6, label=f"{name}: AB")
    axes[2].plot(CS["y_grid"], r["rho_star_NRMF"], "s--", color=colors[name],
                 ms=5, lw=1.1, alpha=0.6, label=f"{name}: NRMF")
axes[2].axvspan(2.8, 3.5, color="#eeeeee", zorder=0)
axes[2].set_xlabel(r"$y=\lambda_\Delta\langle k_\Delta\rangle$")
axes[2].set_ylabel(r"$\rho^*$")
axes[2].set_title(r"(c)", loc="left", fontweight="bold"); axes[2].set_ylim(0, 0.3)
axes[2].legend(loc="lower right", fontsize=7.5)
fig.tight_layout()
save_vector(fig, "fig_combined_3")

# ============================================================== Fig 4 =====
S = load("structural_stats.json")
FS2 = load("finite_size_stats.json")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
combos = [("Poisson", "gNone_p0.0", "#1f77b4"),
          (r"$\gamma=3.5$", "g3.5_pNone", "#2ca02c"),
          (r"$\gamma=2.8$", "g2.8_pNone", "#d62728"),
          (r"$p=0.4$", "gNone_p0.4", "#ff7f0e")]
xp = np.arange(len(combos))
abm, abs_, nmv = [], [], []
for i, (lab, key, _) in enumerate(combos):
    vals = [v for n, x in S["ab_rho_star"][key].items() for v in x["values"]]
    abm.append(np.mean(vals)); abs_.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
    nmv.append(np.mean([v for n, x in S["nrmf_rho_star"][key].items() for v in x["values"]]))
ax1.errorbar(xp, abm, yerr=abs_, fmt="o", ms=9, color="#d62728", capsize=5,
             lw=2.0, label="agent-based")
ax1.plot(xp, nmv, "s", ms=9, color="#1f77b4", lw=2.0, label="NRMF")
ax1.axhline(FS2["fit"]["A"], color="#7f7f7f", ls=":", lw=1.8,
            label=r"MF $\rho^*=0.124$")
ax1.set_xticks(xp); ax1.set_xticklabels([c[0] for c in combos])
ax1.set_ylim(0, 0.28); ax1.set_ylabel(r"critical mass $\rho^*$")
ax1.set_title(r"(a)", loc="left", fontweight="bold")
ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

lg2 = np.linspace(0.005, 0.16, 40)
palo = [pa.integrate(0.005, lam=lm) for lm in lg2]
pahi = [pa.integrate(0.95, lam=lm) for lm in lg2]
ax2.plot(lg2, palo, color="#1f77b4", lw=2.0, label="pair approx low")
ax2.plot(lg2, pahi, color="#1f77b4", ls="--", lw=2.0, label="pair approx high")
# MF reference (same as pair_approx.py)
def mf_rho(lam, r0):
    r = r0
    for _ in range(20000):
        f = -r + 0.6*10*r*(1-r) + 0.6*10*r*r*(1-r)
        r = np.clip(r + 0.005*f, 0, 1)
    return r
ax2.plot(lg2, [mf_rho(l, 0.005) for l in lg2], color="#7f7f7f", lw=2.0,
         label="MF low")
ax2.plot(lg2, [mf_rho(l, 0.95) for l in lg2], color="#7f7f7f", ls="--", lw=2.0,
         label="MF high")
ax2.axvline(0.1, color="#d62728", ls=":", lw=1.5, label=r"$\lambda_c=1/\langle k\rangle$")
ax2.set_xlabel(r"$\lambda$"); ax2.set_ylabel(r"$\rho_\infty$")
ax2.set_title(r"(b)", loc="left", fontweight="bold")
ax2.set_ylim(-0.02, 1.02)
ax2.legend(loc="lower right", fontsize=8, framealpha=0.92)
fig.tight_layout()
save_vector(fig, "fig_combined_4")

# ============================================================== Fig 5 =====
IV = load("intervention_stats.json")
RV = load("recovery_stats.json")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
scol = {"random": ("#7f7f7f", "o"), "degree": ("#1f77b4", "s"),
        "simplex": ("#d62728", "^")}
fs = IV["f_values"]
for strat, (c, mk) in scol.items():
    ys = [IV["results"]["IEEE118"][strat][str(f)]["rho_star"] for f in fs]
    ax1.plot(fs, ys, marker=mk, ms=7, lw=1.8, color=c,
             label=f"{strat} (IEEE 118)")
ax1.plot(fs, [IV["results"]["ER(118)"]["degree"][str(f)]["rho_star"] for f in fs],
         marker="s", ms=5, lw=1.2, color="#1f77b4", ls="--", alpha=0.6,
         label="degree (ER)")
ax1.plot(fs, [IV["results"]["BA(118)"]["degree"][str(f)]["rho_star"] for f in fs],
         marker="s", ms=5, lw=1.2, color="#1f77b4", ls="-.", alpha=0.6,
         label="degree (BA)")
ax1.set_xlabel(r"hardened fraction $f$"); ax1.set_ylabel(r"$\rho^*(f)$")
ax1.set_title(r"(a)", loc="left", fontweight="bold")
ax1.set_ylim(0, 1.0); ax1.set_xticks([0.0, 0.05, 0.10, 0.15])
ax1.legend(loc="upper left", fontsize=8, framealpha=0.92)

mcol = {"const": ("#1f77b4", "o"), "pow": ("#d62728", "s"), "thr": ("#2ca02c", "^")}
mlab = {"const": r"const $\mu=1$", "pow": r"$\mu_i\propto(m_i/k_i)$",
        "thr": r"$\mu_i\propto\Theta(m_i/k_i-0.5)$"}
yg = RV["y_grid"]
for mode, (c, mk) in mcol.items():
    ax2.plot(yg, RV["results"]["IEEE118"][mode]["rho_star"], marker=mk, ms=7,
             lw=1.8, color=c, label=mlab[mode])
for nm, lsname in [("ER(118)", "--"), ("BA(118)", "-.")]:
    ax2.plot(yg, RV["results"][nm]["const"]["rho_star"], marker="o", ms=4,
             lw=1.1, color="#1f77b4", ls=lsname, alpha=0.5, label=f"const ({nm})")
ax2.axvspan(3.0, 4.2, color="#eeeeee", zorder=0)
ax2.set_xlabel(r"$y=\lambda_\Delta\langle k_\Delta\rangle$"); ax2.set_ylabel(r"$\rho^*$")
ax2.set_title(r"(b)", loc="left", fontweight="bold")
ax2.set_ylim(0, 1.0)
ax2.legend(loc="lower right", fontsize=7.5, framealpha=0.92)
fig.tight_layout()
save_vector(fig, "fig_combined_5")

print("ALL VECTOR FIGURES DONE")
