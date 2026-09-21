# -*- coding: utf-8 -*-
"""Direction 2 -- structural sensitivity of the critical mass in simplicial contagion.

Systematically vary:
  (i)   hyperdegree heterogeneity  gamma (Poisson baseline vs power-law config model),
  (ii)  hyperedge overlap p (fraction of triangles sharing an edge with the graph layer),
and measure, for each structure:
  * bistability window at lambda_D<k_D> = 6 (ABM forward/backward at two representative lambda),
  * agent-based critical mass rho*(N) at N in {1000, 5000} with (lambda<k>, lambda_D<k_D>) = (0.4, 6),
  * network-resolved rate-equation critical mass on the same instances.

All stochastic results are written to figures/structural_stats.json.
Runtime ~1.5-2.5 h on an ordinary PC (multiprocessing).
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simplicial_contagion as sc

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# ---- global experiment parameters (keep lambda<k>=0.4, lambda_D<k_D>=6 as baseline) ----
KBAR = 10.0
KDBAR = 10.0
LAM_K = 0.4
LAMD_KD = 6.0
T_MAX = 300.0
DT = 0.02
REALIZ = 8
SURVIVE = 5
N_BISECT = 6
LO, HI = 0.05, 0.30
INSTANCES = {1000: 2, 5000: 2}

GAMMAS = [3.5, 2.8]          # power-law exponents of the hyperdegree (Poisson baseline = existing)
OVERLAPS = [0.0, 0.4]        # fraction of graph edges carrying an enhanced triangle


# ---------------------------------------------------------------------------
# structure generators
# ---------------------------------------------------------------------------
def powerlaw_sequence(n, gamma, kmin=1, kmax=None, target_mean=KDBAR, rng=None):
    """Degree sequence of a bounded power law P(k) ~ k^{-gamma} with the given
    mean, realised by rejection sampling. Returns integer degrees."""
    if rng is None:
        rng = np.random.default_rng()
    if kmax is None:
        kmax = int(n ** 0.5) + 1
    kmin = int(kmin)
    kmax = max(kmax, kmin + 1)
    norm = sum(k ** (-gamma) for k in range(kmin, kmax + 1))
    seq = np.empty(n, dtype=int)
    for i in range(n):
        while True:
            u = rng.random()
            k = int(kmin * (1.0 / (1.0 - u)) ** (1.0 / (gamma - 1.0))) if gamma > 1 else kmin
            k = min(max(k, kmin), kmax)
            if rng.random() < (k ** (-gamma)) / norm:
                seq[i] = k
                break
    return seq


def renormalise_trisum(seq, rng=None):
    """Adjust the degree sequence so its sum is divisible by 3 (minimal change)."""
    s = int(seq.sum())
    r = s % 3
    if r == 0:
        return seq
    seq = seq.copy()
    # add/subtract 1 on a random entry
    i = rng.integers(0, seq.size)
    if seq[i] + (3 - r) <= 200:
        seq[i] += (3 - r)
    else:
        seq[i] -= r
    return seq


def generate_powerlaw_simplicial_complex(n, gamma, rng=None):
    """ER pairwise layer (as baseline) + power-law hyperdegree 2-simplex layer
    built by stub matching. Mean simplex degree ~ KDBAR."""
    if rng is None:
        rng = np.random.default_rng()
    from scipy import sparse

    # pairwise layer identical to baseline
    p = KBAR / (n - 1)
    iu, ju = np.triu_indices(n, 1)
    keep = rng.random(iu.size) < p
    rows = np.concatenate([iu[keep], ju[keep]])
    cols = np.concatenate([ju[keep], iu[keep]])
    adj = sparse.csr_matrix((np.ones(rows.size, dtype=bool), (rows, cols)), shape=(n, n))

    # power-law hyperdegrees
    seq = powerlaw_sequence(n, gamma, rng=rng)
    seq = renormalise_trisum(seq, rng)
    stubs = []
    for node, deg in enumerate(seq):
        stubs.extend([node] * int(deg))
    rng.shuffle(stubs)

    seen = set()
    tris = []
    guard = 0
    while len(stubs) >= 3 and guard < 10 * len(stubs):
        guard += 1
        a, b, c = stubs.pop(), stubs.pop(), stubs.pop()
        t = tuple(sorted((a, b, c)))
        if len(set(t)) == 3 and t not in seen:
            seen.add(t)
            tris.append(t)
        else:
            stubs += [a, b, c]
    tris = np.asarray(tris, dtype=np.int64)
    # normalise: if the stub matching ended with leftover or overshoot, pad with
    # random triples until the mean simplex degree is close to KDBAR
    cur = tris.shape[0] if tris.size else 0
    target = int(round(n * KDBAR / 3.0))
    while tris.shape[0] < target and tris.shape[0] < n * n * n:
        t = tuple(sorted(rng.choice(n, size=3, replace=False)))
        if t not in seen:
            seen.add(t)
            tris = np.vstack([tris, np.array(t, dtype=np.int64)])
    return adj, tris


def generate_overlap_simplicial_complex(n, p_overlap, rng=None):
    """ER layer + triangles; a fraction p_overlap of the graph edges each carry
    one triangle built on that edge (two endpoints + a random third node).
    The remaining triangles are random triples, total mean simplex degree ~KDBAR."""
    if rng is None:
        rng = np.random.default_rng()
    from scipy import sparse

    p = KBAR / (n - 1)
    iu, ju = np.triu_indices(n, 1)
    keep = rng.random(iu.size) < p
    e_i = iu[keep]
    e_j = ju[keep]
    rows = np.concatenate([e_i, e_j])
    cols = np.concatenate([e_j, e_i])
    adj = sparse.csr_matrix((np.ones(rows.size, dtype=bool), (rows, cols)), shape=(n, n))

    target = int(round(n * KDBAR / 3.0))
    seen = set()
    tris = []
    # edge-enhanced triangles
    n_edges = e_i.size
    n_enh = int(round(p_overlap * n_edges))
    if n_enh > 0:
        pick = rng.choice(n_edges, size=n_enh, replace=False)
        for idx in pick:
            a, b = e_i[idx], e_j[idx]
            c = rng.choice(n, size=1)[0]
            while c == a or c == b:
                c = rng.choice(n, size=1)[0]
            t = tuple(sorted((int(a), int(b), int(c))))
            if t not in seen:
                seen.add(t)
                tris.append(t)
    # random padding
    guard = 0
    while len(tris) < target and guard < 10 * target + 1000:
        guard += 1
        t = tuple(sorted(rng.choice(n, size=3, replace=False)))
        if t not in seen:
            seen.add(t)
            tris.append(t)
    return adj, np.asarray(tris, dtype=np.int64)


def _combo_seed(base, n, inst, gamma, p):
    g = 0 if gamma is None else int(gamma * 10)
    pp = 0 if p is None else int(p * 100)
    return base + 10000 * inst + 1000 * n + g * 10 + pp


# ---------------------------------------------------------------------------
# measurements (module level for multiprocessing)
# ---------------------------------------------------------------------------
def _rho_final(cfg):
    adj, tris, lam, lamd, rho0, seed = cfg
    ts, rhos = sc.simulate(adj, tris, lam, lamd, rho0, t_max=T_MAX, dt=DT,
                           rng=np.random.default_rng(seed))
    return float(rhos[-1])


def _ab_critical(args):
    """Bisect agent-based critical mass on one instance. args=(n, gamma, p, inst)."""
    n, gamma, p, inst = args
    rng = np.random.default_rng(_combo_seed(100000, n, inst, gamma, p))
    if gamma is None:
        adj, tris = sc.generate_simplicial_complex(n, KBAR, KDBAR, rng)
    elif p is None:
        adj, tris = generate_powerlaw_simplicial_complex(n, gamma, rng)
    else:
        adj, tris = generate_overlap_simplicial_complex(n, p, rng)
    kbar = float(adj.sum(axis=1).mean())
    kdbar = float(np.bincount(tris.reshape(-1), minlength=n).mean())
    lam = LAM_K / kbar
    lamd = LAMD_KD / kdbar

    lo, hi = LO, HI
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        ok = 0
        for rep in range(REALIZ):
            if _rho_final((adj, tris, lam, lamd, mid, 1000 * rep + 7)) > 0.5:
                ok += 1
        if ok >= SURVIVE:
            hi = mid
        else:
            lo = mid
    rho_star = 0.5 * (lo + hi)
    ok = 0
    for rep in range(REALIZ):
        if _rho_final((adj, tris, lam, lamd, rho_star, 1000 * rep + 7)) > 0.5:
            ok += 1
    return {"N": n, "gamma": gamma, "p": p, "instance": inst,
            "kbar": kbar, "kdbar": kdbar, "rho_star": float(rho_star),
            "survivors": ok}


def nrmf_final(adj, tris, rho0, lam, lamd, steps=40000):
    n = adj.shape[0]
    rho = np.full(n, rho0, dtype=float)
    tri0, tri1, tri2 = tris[:, 0], tris[:, 1], tris[:, 2]
    for _ in range(steps):
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


def _nrmf_critical(args):
    n, gamma, p, inst = args
    rng = np.random.default_rng(_combo_seed(200000, n, inst, gamma, p))
    if gamma is None:
        adj, tris = sc.generate_simplicial_complex(n, KBAR, KDBAR, rng)
    elif p is None:
        adj, tris = generate_powerlaw_simplicial_complex(n, gamma, rng)
    else:
        adj, tris = generate_overlap_simplicial_complex(n, p, rng)
    kbar = float(adj.sum(axis=1).mean())
    kdbar = float(np.bincount(tris.reshape(-1), minlength=n).mean())
    lam = LAM_K / kbar
    lamd = LAMD_KD / kdbar
    r_high = nrmf_final(adj, tris, 0.9, lam, lamd)
    thr = 0.5 * r_high
    lo, hi = LO, HI
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        if nrmf_final(adj, tris, mid, lam, lamd) > thr:
            hi = mid
        else:
            lo = mid
    return {"N": n, "gamma": gamma, "p": p, "instance": inst,
            "rho_star": float(0.5 * (lo + hi)), "r_high": float(r_high)}


def _bistability_check(args):
    """ABM forward/backward density difference at two lambda inside the mean-field
    bistable window (lambda_D<k_D>=6). Returns (lam, rho_fwd, rho_back)."""
    n, gamma, p, lam_vals = args
    rng = np.random.default_rng(_combo_seed(70000, n, 0, gamma, p))
    if gamma is None:
        adj, tris = sc.generate_simplicial_complex(n, KBAR, KDBAR, rng)
    elif p is None:
        adj, tris = generate_powerlaw_simplicial_complex(n, gamma, rng)
    else:
        adj, tris = generate_overlap_simplicial_complex(n, p, rng)
    kbar = float(adj.sum(axis=1).mean())
    kdbar = float(np.bincount(tris.reshape(-1), minlength=n).mean())
    lamd = LAMD_KD / kdbar
    out = []
    for lam in lam_vals:
        fwd = np.mean([sc.steady_state(adj, tris, lam, lamd, 0.001, t_max=200.0, dt=DT,
                                       rng=np.random.default_rng(31 + i)) for i in range(5)])
        back = np.mean([sc.steady_state(adj, tris, lam, lamd, 0.95, t_max=200.0, dt=DT,
                                        rng=np.random.default_rng(61 + i)) for i in range(5)])
        out.append((float(lam), float(fwd), float(back)))
    return out


# ---------------------------------------------------------------------------
def main():
    combos = []          # (gamma, p): None=Poisson baseline
    for g in GAMMAS:
        combos.append((g, None))
    for p in OVERLAPS:
        combos.append((None, p))

    # ---- 1) bistability spot checks (mean-field window) ----
    # mean-field saddle-node window at lambda_D<k_D>=6: solve -r + x r(1-r) + 6 r^2(1-r)=0
    # bistable for lambda<k> in (lambda_sn, 1); sample two interior points
    lam_checks = [0.03, 0.06]
    bi_results = {}
    for gamma, p in combos:
        key = f"g{gamma}_p{p}"
        rng = np.random.default_rng(123 + int(gamma * 10) if gamma else 456 + int(p * 100))
        out = _bistability_check((2000, gamma, p, lam_checks))
        bi_results[key] = [{"lam": l, "rho_fwd": f, "rho_back": b} for l, f, b in out]
        print(f"[struct] {key}: bistability checks done", flush=True)

    # ---- 2) critical mass vs N (ABM + NRMF) ----
    tasks = [(n, gamma, p, i) for (gamma, p) in combos
             for n, cnt in INSTANCES.items() for i in range(cnt)]
    nproc = max(1, (os.cpu_count() or 4) - 1)
    print(f"[struct] {len(tasks)} critical-mass tasks, {nproc} procs", flush=True)

    ab_res, nm_res = [], []
    with Pool(processes=nproc) as pool:
        for r in pool.imap_unordered(_ab_critical, tasks):
            ab_res.append(r)
            print(f"[struct] AB N={r['N']} g={r['gamma']} p={r['p']} "
                  f"inst={r['instance']} rho*={r['rho_star']:.4f} ({r['survivors']}/8)", flush=True)
        for r in pool.imap_unordered(_nrmf_critical, tasks):
            nm_res.append(r)
            print(f"[struct] NRMF N={r['N']} g={r['gamma']} p={r['p']} "
                  f"inst={r['instance']} rho*={r['rho_star']:.4f}", flush=True)

    # aggregate
    def agg(rows):
        d = {}
        for r in rows:
            key = f"g{r['gamma']}_p{r['p']}"
            d.setdefault(key, {}).setdefault(r["N"], []).append(r["rho_star"])
        return {k: {int(n): {"values": v, "mean": float(np.mean(v)),
                             "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                    for n, v in d[k].items()} for k in d}

    data = {
        "params": {"kbar": KBAR, "kdbar": KDBAR, "lam_k": LAM_K, "lamd_kd": LAMD_KD,
                   "t_max": T_MAX, "realiz": REALIZ, "bisect": N_BISECT,
                   "instances": INSTANCES},
        "combos": [{"gamma": g, "p": p} for g, p in combos],
        "bistability": bi_results,
        "ab_rho_star": agg(ab_res),
        "nrmf_rho_star": agg(nm_res),
        "details_ab": ab_res, "details_nrmf": nm_res,
    }
    with open(os.path.join(OUT, "structural_stats.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("[struct] saved structural_stats.json")
    print("[struct] ALL DONE")


if __name__ == "__main__":
    main()
