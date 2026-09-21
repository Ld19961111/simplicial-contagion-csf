"""Survival-probability ramp at N=2000: fraction of realizations that reach
the endemic state (rho > 0.5) as a function of the seed density rho0.
This quantifies how sharp the observable critical mass is."""

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simplicial_contagion as sc

OUT = os.path.join(HERE, "figures")

LAM, LAMD = 0.04, 0.6
T_MAX = 400.0
DT = 0.02
N = 2000
REALIZ = 25
SEEDS = [0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20]


def survival_prob(args):
    rho0, = args
    rng_net = np.random.default_rng(2100000)   # same N=2000, inst=0 network
    adj, tris = sc.generate_simplicial_complex(N, 10, 10, rng_net)
    ok = 0
    for rep in range(REALIZ):
        ts, rhos = sc.simulate(adj, tris, LAM, LAMD, rho0, t_max=T_MAX, dt=DT,
                               rng=np.random.default_rng(1000 * rep + 7))
        if rhos[-1] > 0.5:
            ok += 1
    return {"rho0": rho0, "survivors": ok, "total": REALIZ,
            "prob": ok / REALIZ}


def main():
    tasks = [(s,) for s in SEEDS]
    nproc = max(1, (os.cpu_count() or 4) - 1)
    print(f"[ramp] {len(tasks)} tasks, {nproc} processes", flush=True)
    results = []
    with Pool(processes=nproc) as pool:
        for r in pool.imap_unordered(survival_prob, tasks):
            results.append(r)
            print(f"[ramp] rho0={r['rho0']:.2f}: {r['survivors']}/{r['total']} "
                  f"-> P={r['prob']:.2f}", flush=True)

    results.sort(key=lambda r: r["rho0"])
    with open(os.path.join(OUT, "survival_ramp.json"), "w", encoding="utf-8") as f:
        json.dump({"N": N, "lam": LAM, "lamD": LAMD, "t_max": T_MAX,
                   "realiz": REALIZ, "results": results}, f, indent=2)
    print("[ramp] saved survival_ramp.json", flush=True)
    print("[ramp] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
