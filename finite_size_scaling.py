"""Finite-size scaling of the critical mass in simplicial contagion.

For N in {500, 1000, 2000, 5000, 10000} and several independent network
instances per size we locate the agent-based critical mass rho*(N) at
(lam, lamD) = (0.04, 0.6) by bisection: a seed density is deemed "surviving"
if at least 5 out of 8 stochastic realizations reach rho > 0.5 by t = 300.
We then fit rho*(N) = A + B N^{-alpha} and produce Fig. 4.

Runtime on an ordinary multi-core PC: ~15-30 minutes (multiprocessing).
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

LAM = 0.04
LAMD = 0.6
T_MAX = 300.0
DT = 0.02
REALIZ = 8        # stochastic realizations per bisection step
SURVIVE = 5       # minimum number of survivors to call a seed "surviving"
N_BISECT = 6      # bisection levels -> resolution (0.30-0.05)/2^6 ~ 0.0039
LO, HI = 0.05, 0.30
INSTANCES = {500: 5, 1000: 5, 2000: 5, 5000: 5, 10000: 3}


def survive_fraction(adj, tris, rho0):
    """Fraction of realizations that reach the endemic state by t_max."""
    ok = 0
    for rep in range(REALIZ):
        ts, rhos = sc.simulate(adj, tris, LAM, LAMD, rho0, t_max=T_MAX, dt=DT,
                               rng=np.random.default_rng(1000 * rep + 7))
        if rhos[-1] > 0.5:
            ok += 1
    return ok


def critical_mass_for_instance(args):
    """Bisect the critical mass on one network instance. args = (N, inst_idx)."""
    n, idx = args
    seed = 100000 + 1000 * n + idx
    rng = np.random.default_rng(seed)
    adj, tris = sc.generate_simplicial_complex(n, 10, 10, rng)

    lo, hi = LO, HI
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        if survive_fraction(adj, tris, mid) >= SURVIVE:
            hi = mid
        else:
            lo = mid
    rho_star = 0.5 * (lo + hi)

    # final verification at the reported value
    surv = survive_fraction(adj, tris, rho_star)
    return {"N": n, "instance": idx, "rho_star": float(rho_star),
            "survivors_at_rho_star": surv}


def main():
    tasks = [(n, i) for n, cnt in INSTANCES.items() for i in range(cnt)]
    nproc = max(1, (os.cpu_count() or 4) - 1)
    print(f"[fs] {len(tasks)} tasks, {nproc} processes", flush=True)

    results = []
    with Pool(processes=nproc) as pool:
        for i, r in enumerate(pool.imap_unordered(critical_mass_for_instance, tasks)):
            results.append(r)
            print(f"[fs] {i+1}/{len(tasks)} done: N={r['N']} inst={r['instance']} "
                  f"rho*={r['rho_star']:.4f} (survivors {r['survivors_at_rho_star']}/8)",
                  flush=True)

    # aggregate per size
    per_N = {}
    for r in results:
        per_N.setdefault(r["N"], []).append(r["rho_star"])
    summary = {}
    for n in sorted(per_N):
        arr = np.asarray(per_N[n])
        summary[str(n)] = {
            "rho_star_values": [float(x) for x in arr],
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        }

    # fit rho*(N) = A + B N^{-alpha}
    Ns = np.array(sorted(per_N.keys()), dtype=float)
    means = np.array([summary[str(int(n))]["mean"] for n in Ns])
    stds = np.array([summary[str(int(n))]["std"] for n in Ns])

    fit = None
    try:
        from scipy.optimize import curve_fit
        def law(n, a, b, alpha):
            return a + b * n ** (-alpha)
        popt, pcov = curve_fit(law, Ns, means, p0=[0.124, 0.5, 0.5],
                               sigma=stds + 1e-9, maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        fit = {"A": float(popt[0]), "A_err": float(perr[0]),
               "B": float(popt[1]), "B_err": float(perr[1]),
               "alpha": float(popt[2]), "alpha_err": float(perr[2])}
        print(f"[fs] fit: rho*(N) = {popt[0]:.4f} + {popt[1]:.3f} N^({-popt[2]:.3f})", flush=True)
    except Exception as e:  # pragma: no cover
        print(f"[fs] fit failed: {e}", flush=True)

    # ---- Figure 4 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
        "legend.fontsize": 10, "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    })
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for r in results:
        ax.plot(r["N"], r["rho_star"], "o", color="#1f77b4", ms=5, alpha=0.6)
    ax.errorbar(Ns, means, yerr=stds, fmt="s", color="#d62728", ms=7, capsize=4,
                lw=1.5, label="mean over instances")
    if fit is not None:
        nn = np.linspace(Ns.min(), Ns.max(), 200)
        ax.plot(nn, law(nn, *popt), "--", color="#2ca02c", lw=2.0,
                label=rf"fit: $\rho^*_\infty={popt[0]:.3f},\ \alpha={popt[2]:.2f}$")
    ax.axhline(0.1235, color="#7f7f7f", ls=":", lw=1.5,
               label=r"mean-field $\rho^* = 0.124$")
    ax.set_xscale("log")
    ax.set_xlabel(r"system size  $N$")
    ax.set_ylabel(r"agent-based critical mass  $\rho^*(N)$")
    ax.set_title(rf"Finite-size scaling at $(\lambda,\lambda_\Delta)=(0.04,0.6)$")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_finite_size.png"), dpi=200)
    plt.close(fig)
    print("[fs] fig4 saved", flush=True)

    data = {"params": {"lam": LAM, "lamD": LAMD, "t_max": T_MAX, "dt": DT,
                       "realiz": REALIZ, "survive": SURVIVE,
                       "bisect_levels": N_BISECT, "interval": [LO, HI]},
            "per_N": summary, "fit": fit}
    with open(os.path.join(OUT, "finite_size_stats.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("[fs] saved finite_size_stats.json", flush=True)
    print("[fs] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
