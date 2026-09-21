# -*- coding: utf-8 -*-
"""Direction 4 -- state-dependent recovery coupling in higher-order cascades.

Extends the simplicial cascading-failure model (cascade_infrastructure.py) with
recovery that depends on the local functional-neighbour fraction m_i/k_i:

  mode "const" : mu_i = mu0                      (baseline, mu0=1)
  mode "pow"   : mu_i = mu0 * (m_i/k_i)^alpha    (recovery coupling, Danziger & Barabasi 2022)
  mode "thr"   : mu_i = mu0 * Theta(m_i/k_i > theta)  (threshold/multivariate recovery, Li et al. 2023)

For each mode we measure the critical seed fraction rho*(y) on IEEE 118, ER(118)
and BA(118) across the triadic coupling y, with the pairwise channel fixed at
lam<k> = 0.6. Comparison with the constant-recovery baseline shows how
recovery coupling shifts the self-healing regime (y <= 3.5) and the erosion
regime (y >= 4.5).

Output: figures/recovery_stats.json, figures/fig9_recovery.png
Runtime ~10-25 min on an ordinary PC (multiprocessing).
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simplicial_contagion as sc
import cascade_infrastructure as ci

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

T_MAX = 300.0
DT = 0.02
REALIZ = 15
N_BISECT = 6
LO, HI = 0.02, 0.90
Y_GRID = [3.5, 4.5, 6.0, 8.0, 10.0]
MODES = [("const", None), ("pow", 1.0), ("thr", 0.5)]


def get_topologies():
    rng = np.random.default_rng(2025)
    import networkx as nx
    g118 = nx.Graph()
    g118.add_edges_from(ci.ieee118_edges())
    out = {}
    for name, g, seed in [("IEEE118", g118, 0), ("ER(118)", None, 2025),
                          ("BA(118)", None, 2026)]:
        if name == "IEEE118":
            adj, tris = ci.build(g118, rng)
        elif name.startswith("ER"):
            kbar0 = float(np.array([d for _, d in g118.degree()]).mean())
            gg = nx.erdos_renyi_graph(118, kbar0 / 117, seed=seed)
            adj, tris = ci.build(gg, rng)
        else:
            gg = nx.barabasi_albert_graph(118, 2, seed=seed)
            adj, tris = ci.build(gg, rng)
        kbar = float(adj.sum(axis=1).mean())
        kdbar = float(np.bincount(tris.reshape(-1), minlength=118).mean())
        out[name] = {"adj": adj, "tris": tris, "kbar": kbar, "kdbar": kdbar}
    return out


def simulate_rc(adj, tris, lam, lamd, rho0, mode, param, t_max=T_MAX, dt=DT,
                rng_seed=0):
    """Agent-based simulation with state-dependent recovery."""
    rng = np.random.default_rng(rng_seed)
    n = adj.shape[0]
    inf = rng.random(n) < rho0
    k_i = np.asarray(adj.sum(axis=1)).ravel().astype(float)
    n_steps = int(round(t_max / dt))
    rhos = np.empty(n_steps + 1)
    rhos[0] = inf.mean()
    for step in range(n_steps):
        k_inf = np.asarray(adj @ inf).ravel()
        cnt = inf[tris[:, 0]].astype(np.int64) + inf[tris[:, 1]].astype(np.int64) + inf[tris[:, 2]].astype(np.int64)
        mask2 = cnt == 2
        sus = tris[mask2]
        sus_flat = sus.reshape(-1)
        is_sus = ~inf[sus_flat]
        idx = sus_flat[is_sus]
        k_d_inf = np.bincount(idx, minlength=n)

        rate_inf = lam * k_inf + lamd * k_d_inf
        p_inf = 1.0 - np.exp(-rate_inf * dt)

        if mode == "const":
            mu = np.ones(n)
        elif mode == "pow":
            frac = 1.0 - k_inf / np.maximum(k_i, 1.0)
            mu = frac ** param
        else:  # thr
            frac = 1.0 - k_inf / np.maximum(k_i, 1.0)
            mu = (frac > param).astype(float)

        p_rec = 1.0 - np.exp(-mu * dt)
        new_inf = (~inf) & (rng.random(n) < p_inf)
        new_rec = inf & (rng.random(n) < p_rec)
        inf = (inf | new_inf) & ~new_rec
        rhos[step + 1] = inf.mean()
    return rhos


def final_rho(cfg):
    adj, tris, lam, lamd, rho0, mode, param, seed = cfg
    rhos = simulate_rc(adj, tris, lam, lamd, rho0, mode, param, rng_seed=seed)
    return float(rhos[-1])


def ab_point(args):
    name, y, mode, param, topo = args
    adj, tris = topo["adj"], topo["tris"]
    lam = 0.6 / topo["kbar"]
    lamd = y / topo["kdbar"]

    r_high = float(np.mean([final_rho((adj, tris, lam, lamd, 0.9, mode, param, 1000 * i + 7))
                            for i in range(REALIZ)]))
    thr = 0.5 * r_high
    lo, hi = LO, HI
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        ok = 0
        for rep in range(REALIZ):
            if final_rho((adj, tris, lam, lamd, mid, mode, param, 1000 * rep + 7)) > thr:
                ok += 1
        if ok >= 0.5 * REALIZ:
            hi = mid
        else:
            lo = mid
    rho_star = 0.5 * (lo + hi)
    return {"name": name, "y": y, "mode": mode, "param": param,
            "rho_high": float(r_high), "rho_star": float(rho_star)}


_TOPO = None  # kept for reference; workers receive topo via args


def main():
    _TOPO = get_topologies()
    for name, t in _TOPO.items():
        print(f"[recover] {name}: <k>={t['kbar']:.2f} <kD>={t['kdbar']:.2f}", flush=True)

    jobs = [(name, y, mode, param, _TOPO[name]) for name in _TOPO for y in Y_GRID
            for mode, param in MODES]
    results = {name: {m: {"y": Y_GRID, "rho_star": [], "rho_high": []}
                      for m, _ in MODES} for name in _TOPO}

    nproc = max(1, int(os.environ.get("NPROC", (os.cpu_count() or 4) - 1)))
    print(f"[recover] using {nproc} workers", flush=True)
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        futs = {ex.submit(ab_point, j): j for j in jobs}
        for fut, (name, y, mode, param, _) in futs.items():
            r = fut.result()
            results[name][mode]["rho_star"].append(r["rho_star"])
            results[name][mode]["rho_high"].append(r["rho_high"])
            print(f"[recover] {name} {mode} y={y}: rho*={r['rho_star']:.3f} "
                  f"(high={r['rho_high']:.3f})", flush=True)

    with open(os.path.join(OUT, "recovery_stats.json"), "w", encoding="utf-8") as fh:
        json.dump({"t_max": T_MAX, "realiz": REALIZ, "bisect": N_BISECT,
                   "y_grid": Y_GRID, "modes": MODES, "results": results},
                  fh, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
                         "legend.fontsize": 8.5, "mathtext.fontset": "stix",
                         "font.family": "STIXGeneral"})
    style = {"const": ("#1f77b4", "o-", r"const $\mu=1$"),
             "pow": ("#d62728", "s--", r"coupling $\mu_i\propto(m_i/k_i)$"),
             "thr": ("#2ca02c", "^-.", r"threshold $\mu_i\propto\Theta(m_i/k_i-0.5)$")}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, name in zip(axes, results):
        for mode, (c, mk, lab) in style.items():
            ax.plot(Y_GRID, results[name][mode]["rho_star"], mk, color=c, ms=7,
                    lw=1.8, label=lab)
        ax.set_xlabel(r"triadic coupling $y=\lambda_\Delta\langle k_\Delta\rangle$")
        ax.set_ylabel(r"critical seed fraction $\rho^*$")
        ax.set_title(name, fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig9_recovery.png"), dpi=200)
    plt.close(fig)
    print("[recover] fig9 saved")
    print("[recover] ALL DONE")


if __name__ == "__main__":
    main()
