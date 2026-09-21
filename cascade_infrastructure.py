"""Extension 5.1 -- higher-order cascading failure on infrastructure networks.

Simplicial SIS cascading-failure model: nodes fail pairwise (rate lam) and
triadically (rate lamD when two neighbouring nodes are already failed).
The 2-simplex layer = graph triangles + a controlled set of random triangles
added on top, reaching a fixed mean simplex degree <kD> = 3.0 (M2 = 118 on
N = 118) for every topology.  The random-triangle layer models shared physical
dependencies (common upstream feeder, joint hydraulic pressure zone,
redundant-circuit coupling) that are not encoded in the transmission graph
itself.

Topologies (same size N=118): IEEE 118-bus (MATPOWER case118), Erdos-Renyi
and Barabasi-Albert.  Pairwise channel normalised to lam*<k> = 0.6 (below the
pairwise threshold) so that cascades exist only through triadic coupling.

Robustness metric: the critical seed fraction rho* -- minimal fraction of
initially failed components that triggers a large cascade -- as a function of
the triadic coupling strength y = lamD*<kD>.  "Large cascade" is self
calibrated: rho_high = mean final density from a 90% seed; a seed survives if
its final density exceeds 0.5*rho_high.

Figures
-------
figures/fig5a_topology.png      : IEEE 118-bus network drawing.
figures/fig5b_hysteresis.png    : IEEE 118 failure-fraction hysteresis at y=5.
figures/fig6_critical_mass.png  : critical seed fraction vs y for the three
                                  topologies (agent-based vs rate equations).

Runtime: ~30-60 min on any ordinary PC (multiprocessing, 3 workers).
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

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 10, "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
})

T_MAX = 300.0
DT = 0.02
REALIZ = 30
N_BISECT = 7
LO, HI = 0.02, 0.90
KD_TARGET = 2.0          # mean simplex degree of the added layer
Y_GRID = [2.8, 3.5, 4.5, 6.0, 8.0, 10.0]


# ---------------------------------------------------------------------------
# Topology builders
# ---------------------------------------------------------------------------
def ieee118_edges():
    txt = open(os.path.join(HERE, "case118.m"), encoding="utf-8",
               errors="replace").read()
    m = re.search(r"mpc\.branch\s*=\s*\[(.*?)\];", txt, re.S)
    rows = [ln.split() for ln in m.group(1).splitlines()
            if ln.strip() and not ln.strip().startswith("%")]
    return list({(int(r[0]), int(r[1])) for r in rows if len(r) >= 2})


def triangles_of(g):
    return np.array([list(t) for t in nx.enumerate_all_cliques(g) if len(t) == 3],
                    dtype=np.int64)


def build(adj_nx, rng):
    """adj_nx -> (adj_csr, tris, nodes); adds random triangles to reach
    <kD> = KD_TARGET."""
    nodes = list(adj_nx.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    edges = [(idx[u], idx[v]) for u, v in adj_nx.edges()]
    rows = [u for u, v in edges] + [v for u, v in edges]
    cols = [v for u, v in edges] + [u for u, v in edges]
    from scipy import sparse
    adj = sparse.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)),
                            shape=(n, n))
    tris = triangles_of(adj_nx)
    tris = np.array([[idx[a], idx[b], idx[c]] for a, b, c in tris],
                    dtype=np.int64)
    have = len(tris)
    need = int(round(n * KD_TARGET / 2.0))
    # add random triangles on distinct node triples
    pool = set()
    while len(tris) < max(have, need):
        t = tuple(sorted(rng.choice(n, size=3, replace=False)))
        if t in pool:
            continue
        pool.add(t)
        tris = np.vstack([tris, np.array(t, dtype=np.int64)])
    return adj, tris


# ---------------------------------------------------------------------------
# Measurements (module level for multiprocessing)
# ---------------------------------------------------------------------------
def _final_rho(seed_cfg):
    adj, tris, lam, lamd, rho0 = seed_cfg
    finals = np.empty(REALIZ)
    for rep in range(REALIZ):
        ts, rhos = sc.simulate(adj, tris, lam, lamd, rho0, t_max=T_MAX, dt=DT,
                               rng=np.random.default_rng(1000 * rep + 7))
        finals[rep] = rhos[-1]
    return finals


def _ab_point(cfg):
    adj, tris, lam, lamd = cfg
    r_high = float(np.mean(_final_rho((adj, tris, lam, lamd, 0.9))))
    thr = 0.5 * r_high
    lo, hi = LO, HI
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        finals = _final_rho((adj, tris, lam, lamd, mid))
        if np.mean(finals > thr) >= 0.5:
            hi = mid
        else:
            lo = mid
    return r_high, thr, 0.5 * (lo + hi)


def nrmf_final(adj, tris, rho0, lam, lamd):
    n = adj.shape[0]
    rho = np.full(n, rho0, dtype=float)
    tri0, tri1, tri2 = tris[:, 0], tris[:, 1], tris[:, 2]
    for _ in range(40000):
        s = adj @ rho
        acc = np.zeros(n)
        np.add.at(acc, tri0, rho[tri1] * rho[tri2])
        np.add.at(acc, tri1, rho[tri0] * rho[tri2])
        np.add.at(acc, tri2, rho[tri0] * rho[tri1])
        drho = -rho + (1.0 - rho) * (lam * s + lamd * acc)
        rho = rho + DT * drho
        if np.max(np.abs(drho)) < 1e-10:
            break
    return float(rho.mean())


def _nrmf_point(cfg):
    adj, tris, lam, lamd = cfg
    r_high = nrmf_final(adj, tris, 0.9, lam, lamd)
    thr = 0.5 * r_high
    lo, hi = LO, HI
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        if nrmf_final(adj, tris, mid, lam, lamd) > thr:
            hi = mid
        else:
            lo = mid
    return r_high, thr, 0.5 * (lo + hi)


def sweep_point(adj, tris, lamd, lam, rho0):
    vals = []
    for rep in range(5):
        ts, rhos = sc.simulate(adj, tris, lam, lamd, rho0, t_max=200.0, dt=DT,
                               rng=np.random.default_rng(100 + rep))
        vals.append(rhos[int(0.8 * rhos.size):].mean())
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(2025)

    # ---- topologies ----
    topo = {}
    g118 = nx.Graph()
    g118.add_edges_from(ieee118_edges())
    adj, tris = build(g118, rng)
    kbar = adj.sum(axis=1).mean()
    kdbar = np.bincount(tris.reshape(-1), minlength=adj.shape[0]).mean()
    topo["IEEE118"] = {"adj": adj, "tris": tris, "kbar": kbar, "kdbar": kdbar,
                       "nx": g118}
    print(f"[cascade] IEEE118: N={adj.shape[0]} <k>={kbar:.2f} "
          f"<kD>={kdbar:.2f} M2={tris.shape[0]}")

    g_er = nx.erdos_renyi_graph(118, kbar / 117, seed=2025)
    adj, tris = build(g_er, rng)
    kbar2 = adj.sum(axis=1).mean()
    kdbar2 = np.bincount(tris.reshape(-1), minlength=118).mean()
    topo["ER(118)"] = {"adj": adj, "tris": tris, "kbar": kbar2, "kdbar": kdbar2}
    print(f"[cascade] ER(118): <k>={kbar2:.2f} <kD>={kdbar2:.2f} "
          f"M2={tris.shape[0]}")

    g_ba = nx.barabasi_albert_graph(118, 2, seed=2026)
    adj, tris = build(g_ba, rng)
    kbar3 = adj.sum(axis=1).mean()
    kdbar3 = np.bincount(tris.reshape(-1), minlength=118).mean()
    topo["BA(118)"] = {"adj": adj, "tris": tris, "kbar": kbar3, "kdbar": kdbar3}
    print(f"[cascade] BA(118): <k>={kbar3:.2f} <kD>={kdbar3:.2f} "
          f"M2={tris.shape[0]}")

    # ---- Fig. 5a: IEEE 118 topology ----
    pos = nx.spring_layout(g118, seed=42, k=0.9)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    d = [g118.degree(u) for u in g118]
    nx.draw_networkx_edges(g118, pos, ax=ax, edge_color="#b8c4d8", width=0.7)
    nx.draw_networkx_nodes(g118, pos, ax=ax, node_size=60, node_color=d,
                           cmap="viridis", vmin=min(d), vmax=max(d))
    nx.draw_networkx_labels(g118, pos, ax=ax, font_size=5, font_color="#333333")
    ax.set_title("IEEE 118-bus test network (MATPOWER case118), "
                 "186 branches, 23 triangles")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5a_topology.png"), dpi=200)
    plt.close(fig)
    print("[cascade] fig5a saved")

    # ---- Fig. 5b: hysteresis on IEEE 118 at y=5 ----
    t118 = topo["IEEE118"]
    lamd5 = 5.0 / t118["kdbar"]
    lam_vals = np.linspace(0.0, 0.55, 23)
    fwd, back = [], []
    for lam in lam_vals:                       # forward: small seed
        fwd.append(sweep_point(t118["adj"], t118["tris"], lamd5, lam, 0.01))
    for lam in lam_vals[::-1]:                 # backward: full seed
        back.append(sweep_point(t118["adj"], t118["tris"], lamd5, lam, 0.95))
    back = back[::-1]
    lam_th = np.linspace(0.0, 0.55, 60)
    nrmf_lo = [nrmf_final(t118["adj"], t118["tris"], 0.005, lam, lamd5)
               for lam in lam_th]
    nrmf_hi = [nrmf_final(t118["adj"], t118["tris"], 0.9, lam, lamd5)
               for lam in lam_th]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(lam_th, nrmf_lo, color="#1f77b4", lw=2.0, label="rate equations: low seed")
    ax.plot(lam_th, nrmf_hi, color="#1f77b4", ls="--", lw=2.0,
            label="rate equations: high seed")
    ax.plot(lam_vals, fwd, "o", ms=5, color="#2ca02c", label="agent-based: forward")
    ax.plot(lam_vals, back, "s", ms=5, color="#ff7f0e", label="agent-based: backward")
    ax.axvline(1.0 / t118["kbar"], color="#d62728", ls=":", lw=1.5,
               label=r"pairwise threshold $\lambda_c=1/\langle k\rangle$")
    ax.set_xlabel(r"pairwise failure rate  $\lambda$")
    ax.set_ylabel(r"steady-state failed fraction  $\rho_\infty$")
    ax.set_title(r"IEEE 118: triadic coupling $y=\lambda_\Delta\langle k_\Delta\rangle=5$")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5b_hysteresis.png"), dpi=200)
    plt.close(fig)
    print("[cascade] fig5b saved")

    # ---- Fig. 6: critical seed fraction vs y (multiprocessing) ----
    jobs_ab, jobs_nm = [], []
    for name, t in topo.items():
        lam0 = 0.6 / t["kbar"]
        for y in Y_GRID:
            lamd = y / t["kdbar"]
            cfg = (t["adj"], t["tris"], lam0, lamd)
            jobs_ab.append((name, y, cfg))
            jobs_nm.append((name, y, cfg))

    results = {name: {"lam0": 0.6 / t["kbar"], "y": Y_GRID,
                      "rho_star_AB": [], "rho_star_NRMF": [],
                      "rho_high_AB": [], "rho_high_NRMF": []}
               for name, t in topo.items()}

    with ProcessPoolExecutor(max_workers=3) as ex:
        ab_futs = {ex.submit(_ab_point, cfg): (name, y)
                   for name, y, cfg in jobs_ab}
        nm_futs = {ex.submit(_nrmf_point, cfg): (name, y)
                   for name, y, cfg in jobs_nm}
        for fut, (name, y) in ab_futs.items():
            r_high, thr, r_star = fut.result()
            results[name]["rho_high_AB"].append(r_high)
            results[name]["rho_star_AB"].append(r_star)
            print(f"[cascade] {name}: y={y:.1f} rho*_AB={r_star:.3f} "
                  f"(high={r_high:.3f}, thr={thr:.3f})", flush=True)
        for fut, (name, y) in nm_futs.items():
            r_high, thr, r_star = fut.result()
            results[name]["rho_high_NRMF"].append(r_high)
            results[name]["rho_star_NRMF"].append(r_star)
            print(f"[cascade] {name}: y={y:.1f} rho*_NRMF={r_star:.3f}", flush=True)

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    colors = {"IEEE118": "#d62728", "ER(118)": "#1f77b4", "BA(118)": "#2ca02c"}
    for name, r in results.items():
        ax.plot(Y_GRID, r["rho_star_AB"], "o-", color=colors[name], ms=7, lw=1.8,
                label=f"{name}: agent-based")
        ax.plot(Y_GRID, r["rho_star_NRMF"], "s--", color=colors[name], ms=6,
                lw=1.2, alpha=0.65, label=f"{name}: rate equations")
    ax.set_xlabel(r"triadic coupling strength  $y=\lambda_\Delta\langle k_\Delta\rangle$")
    ax.set_ylabel(r"critical seed fraction  $\rho^*$")
    ax.set_title(r"Grid robustness vs. triadic coupling, "
                 r"$\lambda\langle k\rangle=0.6$, $N=118$")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_critical_mass.png"), dpi=200)
    plt.close(fig)
    print("[cascade] fig6 saved")

    with open(os.path.join(OUT, "cascade_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"t_max": T_MAX, "dt": DT, "realiz": REALIZ,
                   "survive_frac": 0.5, "kd_target": KD_TARGET,
                   "y_grid": Y_GRID, "results": results}, f, indent=2)
    print("[cascade] saved cascade_stats.json")
    print("[cascade] ALL DONE")


if __name__ == "__main__":
    main()
