# -*- coding: utf-8 -*-
"""Direction 3 -- targeted protection vs random hardening in higher-order cascades.

On IEEE 118, ER(118) and BA(118) with the simplicial cascading-failure model of
cascade_infrastructure.py, we harden (remove from the dynamics) a fraction f of
nodes before the cascade and measure the resulting critical seed fraction
rho*(f) on the surviving network, keeping the physical rates of the ORIGINAL
graph (lam = 0.6/<k>_0, lamD = y/<kD>_0, y=6).

Strategies:
  * random      -- uniformly random hardened set,
  * degree      -- top-f nodes by pairwise degree,
  * simplex     -- top-f nodes by triangle participation (hyper-core heuristic).

Output: figures/intervention_stats.json, figures/fig8_intervention.png
Runtime ~20-50 min on an ordinary PC (multiprocessing).
"""
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simplicial_contagion as sc
import cascade_infrastructure as ci

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

Y = 6.0
T_MAX = 300.0
DT = 0.02
REALIZ = 15
N_BISECT = 6
LO, HI = 0.02, 0.90
FS = [0.05, 0.10, 0.15]
STRATS = ["random", "degree", "simplex"]


def get_topologies():
    rng = np.random.default_rng(2025)
    g118 = __import__("networkx").Graph()
    g118.add_edges_from(ci.ieee118_edges())
    out = {}
    for name, g, seed in [("IEEE118", g118, 0), ("ER(118)", None, 2025),
                          ("BA(118)", None, 2026)]:
        if name == "IEEE118":
            adj, tris = ci.build(g118, rng)
        elif name.startswith("ER"):
            kbar0 = float(np.array([d for _, d in g118.degree()]).mean())
            gg = __import__("networkx").erdos_renyi_graph(118, kbar0 / 117, seed=seed)
            adj, tris = ci.build(gg, rng)
        else:
            gg = __import__("networkx").barabasi_albert_graph(118, 2, seed=seed)
            adj, tris = ci.build(gg, rng)
        kbar = float(adj.sum(axis=1).mean())
        kdbar = float(np.bincount(tris.reshape(-1), minlength=118).mean())
        out[name] = {"adj": adj, "tris": tris, "kbar": kbar, "kdbar": kdbar}
    return out


def rank_by(adj, tris, strat):
    n = adj.shape[0]
    deg = np.asarray(adj.sum(axis=1)).ravel()
    tri_cnt = np.bincount(tris.reshape(-1), minlength=n)
    if strat == "degree":
        return np.argsort(-deg, kind="stable")
    if strat == "simplex":
        return np.argsort(-tri_cnt, kind="stable")
    return np.random.default_rng(42).permutation(n)


def remove_nodes(adj, tris, keep):
    """Return (adj2, tris2) restricted to the kept node set (reindexed)."""
    keep = np.asarray(keep, dtype=bool)
    new_id = np.full(adj.shape[0], -1, dtype=np.int64)
    new_id[keep] = np.arange(int(keep.sum()))
    # pairwise edges among kept nodes
    from scipy import sparse
    coo = adj.tocoo()
    mask = keep[coo.row] & keep[coo.col]
    rows = new_id[coo.row[mask]]
    cols = new_id[coo.col[mask]]
    adj2 = sparse.csr_matrix((np.ones(rows.size, dtype=bool), (rows, cols)),
                             shape=(int(keep.sum()), int(keep.sum())))
    # triangles fully inside the kept set
    tm = keep[tris[:, 0]] & keep[tris[:, 1]] & keep[tris[:, 2]]
    tris2 = new_id[tris[tm]]
    return adj2, tris2


def final_rho(adj, tris, lam, lamd, rho0, rng_seed):
    ts, rhos = sc.simulate(adj, tris, lam, lamd, rho0, t_max=T_MAX, dt=DT,
                           rng=np.random.default_rng(rng_seed))
    return float(rhos[-1])


def ab_critical(args):
    name, strat, f, topo = args
    adj0, tris0 = topo["adj"], topo["tris"]
    lam0 = 0.6 / topo["kbar"]
    lamd0 = Y / topo["kdbar"]
    n0 = adj0.shape[0]

    order = rank_by(adj0, tris0, strat)
    n_hard = int(round(f * n0))
    hard = np.zeros(n0, dtype=bool)
    hard[order[:n_hard]] = True
    keep = ~hard
    adj, tris = remove_nodes(adj0, tris0, keep)
    n_rem = adj.shape[0]

    # physical rates stay those of the original graph
    r_high = float(np.mean([final_rho(adj, tris, lam0, lamd0, 0.9, 1000 * i + 7)
                            for i in range(REALIZ)]))
    thr = 0.5 * r_high
    lo, hi = LO, HI
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        ok = 0
        for rep in range(REALIZ):
            if final_rho(adj, tris, lam0, lamd0, mid, 1000 * rep + 7) > thr:
                ok += 1
        if ok >= 0.5 * REALIZ:
            hi = mid
        else:
            lo = mid
    rho_star = 0.5 * (lo + hi)
    return {"name": name, "strat": strat, "f": f, "n_hardened": n_hard,
            "n_remaining": n_rem, "r_high": float(r_high),
            "rho_star": float(rho_star)}


_TOPO = None


def main():
    _TOPO = get_topologies()
    for name, t in _TOPO.items():
        print(f"[interv] {name}: <k>={t['kbar']:.2f} <kD>={t['kdbar']:.2f} "
              f"M2={t['tris'].shape[0]}", flush=True)

    jobs = [(name, strat, f, _TOPO[name]) for name in _TOPO for strat in STRATS for f in FS]
    results = {name: {s: {} for s in STRATS} for name in _TOPO}

    nproc = max(1, int(os.environ.get("NPROC", (os.cpu_count() or 4) - 1)))
    print(f"[interv] using {nproc} workers", flush=True)
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        futs = {ex.submit(ab_critical, j): j for j in jobs}
        for fut, (name, strat, f, _) in futs.items():
            r = fut.result()
            results[name][strat][f] = r
            print(f"[interv] {name} {strat} f={f}: rho*={r['rho_star']:.3f} "
                  f"(rem={r['n_remaining']})", flush=True)

    with open(os.path.join(OUT, "intervention_stats.json"), "w", encoding="utf-8") as fh:
        json.dump({"y": Y, "t_max": T_MAX, "realiz": REALIZ, "bisect": N_BISECT,
                   "f_values": FS, "strategies": STRATS, "results": results},
                  fh, indent=2)

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
                         "legend.fontsize": 9, "mathtext.fontset": "stix",
                         "font.family": "STIXGeneral"})
    colors = {"random": "#7f7f7f", "degree": "#1f77b4", "simplex": "#d62728"}
    marks = {"random": "o", "degree": "s", "simplex": "^"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, name in zip(axes, results):
        for strat in STRATS:
            fs = sorted(results[name][strat].keys())
            ys = [results[name][strat][f]["rho_star"] for f in fs]
            ax.plot(fs, ys, marker=marks[strat], ms=7, lw=1.8, color=colors[strat],
                    label=strat)
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_xlabel(r"hardened fraction $f$")
        ax.set_ylabel(r"critical seed fraction $\rho^*(f)$")
        ax.set_title(name, fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig8_intervention.png"), dpi=200)
    plt.close(fig)
    print("[interv] fig8 saved")
    print("[interv] ALL DONE")


if __name__ == "__main__":
    main()
