"""Simplicial models of social contagion -- reproduction of Iacopini, Petri, Barrat & Latora,
Nature Communications 10, 2485 (2019).

Implements:
  * generation of a random simplicial complex (ER graph + random 2-simplices),
  * continuous-time SIS-like contagion with pairwise and 2-simplex infection channels,
  * mean-field fixed-point analysis (S-curve, bistability, phase diagram),
  * discrete-time agent-based simulation with exact exponential probabilities.

Model equations (homogeneous mean field, Iacopini et al. 2019, Eq. 3):
    drho/dt = -rho + lam <k> rho (1-rho) + lamD <kD> rho^2 (1-rho)
where rho is the infected fraction, lam the pairwise infection rate,
lamD the 2-simplex infection rate, <k> mean pairwise degree, <kD> mean
number of incident 2-simplices. Recovery rate is set to 1.
"""

import numpy as np


def generate_simplicial_complex(n, kbar, kdbar, rng=None):
    """Build a random simplicial complex.

    - Pairwise links: Erdos-Renyi G(n, p) with p = kbar / (n - 1).
    - 2-simplices: ~ kdbar*n/3 random triples of nodes.

    Returns
    -------
    adj : scipy.sparse.csr_matrix, boolean adjacency of the pairwise graph.
    tris : np.ndarray, shape (M, 3), rows are node triples (2-simplices).
    """
    from scipy import sparse

    if rng is None:
        rng = np.random.default_rng()

    # ---- pairwise layer (ER) ----
    p = kbar / (n - 1)
    iu, ju = np.triu_indices(n, 1)
    keep = rng.random(iu.size) < p
    rows = np.concatenate([iu[keep], ju[keep]])
    cols = np.concatenate([ju[keep], iu[keep]])
    data = np.ones(rows.size, dtype=bool)
    adj = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    # ---- 2-simplex layer (random triples) ----
    m_target = int(round(kdbar * n / 3.0))
    max_triples = n * (n - 1) * (n - 2) // 6
    m_target = min(m_target, max_triples)
    seen = set()
    tris = []
    guard = 0
    while len(tris) < m_target and guard < 10 * m_target + 1000:
        guard += 1
        t = tuple(sorted(rng.choice(n, size=3, replace=False)))
        if t not in seen:
            seen.add(t)
            tris.append(t)
    return adj, np.asarray(tris, dtype=np.int64)


def simulate(adj, tris, lam, lamd, rho0, t_max=400.0, dt=0.02, rng=None):
    """Agent-based simulation with synchronous (exact-rate) updates.

    Returns
    -------
    ts : np.ndarray  (time points)
    rhos : np.ndarray (infected fraction time series)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = adj.shape[0]
    inf = rng.random(n) < rho0
    n_steps = int(round(t_max / dt))
    ts = np.arange(n_steps + 1) * dt
    rhos = np.empty(n_steps + 1, dtype=float)
    rhos[0] = inf.mean()

    for step in range(n_steps):
        # pairwise infected neighbours
        k_i = adj @ inf
        # 2-simplex channel: triangles with exactly two infected members
        cnt = inf[tris[:, 0]].astype(np.int64) + inf[tris[:, 1]].astype(np.int64) + inf[tris[:, 2]].astype(np.int64)
        mask2 = cnt == 2
        sus = tris[mask2]                       # (M2, 3) triangles with 2 infected
        sus_flat = sus.reshape(-1)
        is_sus = ~inf[sus_flat]
        idx = sus_flat[is_sus]                  # the susceptible member of each such triangle
        k_d_inf = np.bincount(idx, minlength=n)

        rate_inf = lam * k_i + lamd * k_d_inf
        p_inf = 1.0 - np.exp(-rate_inf * dt)
        p_rec = 1.0 - np.exp(-dt)

        new_inf = (~inf) & (rng.random(n) < p_inf)
        new_rec = inf & (rng.random(n) < p_rec)
        inf = inf | new_inf
        inf = inf & ~new_rec
        rhos[step + 1] = inf.mean()

    return ts, rhos


def steady_state(adj, tris, lam, lamd, rho0, t_max=400.0, dt=0.02, rng=None):
    """Run simulation and return the time-averaged steady-state density."""
    ts, rhos = simulate(adj, tris, lam, lamd, rho0, t_max, dt, rng)
    tail = rhos[int(0.8 * rhos.size):]
    return tail.mean()


# --------------------------------------------------------------------------
# Mean-field analysis
# --------------------------------------------------------------------------

def mf_fixed_points(lam, lamd, kbar, kdbar, rho_grid=None):
    """Stable / unstable fixed points of the homogeneous mean-field equation.

    f(rho) = -rho + lam*kbar*rho*(1-rho) + lamd*kdbar*rho^2*(1-rho)

    Returns (stable, unstable) lists of rho* in [0,1].
    """
    if rho_grid is None:
        rho_grid = np.linspace(0.0, 1.0, 200001)

    def f(r):
        return -r + lam * kbar * r * (1.0 - r) + lamd * kdbar * r * r * (1.0 - r)

    fv = f(rho_grid)
    # find zero crossings of f (sign changes), excluding rho = 0 handled separately
    signs = np.sign(fv)
    cross = np.where(np.diff(signs) != 0)[0]
    roots = []
    for c in cross:
        # linear interpolation between grid points c and c+1
        r0, r1 = rho_grid[c], rho_grid[c + 1]
        f0, f1 = fv[c], fv[c + 1]
        roots.append(r0 - f0 * (r1 - r0) / (f1 - f0))
    roots = [r for r in roots if 1e-9 < r < 1.0 - 1e-9]

    # stability of a root: f'(r) < 0 -> stable
    stable, unstable = [], []
    for r in roots:
        fp = -1 + lam * kbar * (1.0 - 2.0 * r) + lamd * kdbar * (2.0 * r - 3.0 * r * r)
        (stable if fp < 0 else unstable).append(r)

    # rho = 0 fixed point: stable if f'(0) = lam*kbar - 1 < 0
    if lam * kbar - 1.0 < 0:
        stable.append(0.0)
    else:
        unstable.append(0.0)
    return sorted(stable), sorted(unstable)


def is_bistable(lam, lamd, kbar, kdbar):
    """True when rho=0 is stable AND a positive stable fixed point exists."""
    stable, _ = mf_fixed_points(lam, lamd, kbar, kdbar)
    return (0.0 in stable) and any(s > 1e-6 for s in stable)


def sweep_hysteresis(adj, tris, kbar, kdbar, lam_vals, lamd, t_max=400.0, dt=0.02, rng=None):
    """Forward (small seed, increasing lambda) and backward (large seed,
    decreasing lambda) sweeps.  Each step starts from the previous steady state,
    which is what produces the hysteresis loop."""
    if rng is None:
        rng = np.random.default_rng()
    n = adj.shape[0]

    rng_f = np.random.default_rng(20260101)
    rng_b = np.random.default_rng(20260202)

    # backward: start fully infected at the largest lambda
    lam_asc = np.sort(lam_vals)
    lam_desc = lam_asc[::-1]

    back = []
    state = np.ones(n, dtype=bool)
    for lam in lam_desc:
        ts, rhos = simulate(adj, tris, lam, lamd, state.mean(), t_max, dt, rng=rng_b)
        rho_inf = rhos[int(0.8 * rhos.size):].mean()
        back.append(rho_inf)
        state = np.random.default_rng(seed=None).random(n) < rho_inf  # resample at density

    # forward: start nearly empty at the smallest lambda
    fwd = []
    state = np.zeros(n, dtype=bool)
    state[rng_f.choice(n, size=max(1, int(0.001 * n)), replace=False)] = True
    for lam in lam_asc:
        ts, rhos = simulate(adj, tris, lam, lamd, state.mean(), t_max, dt, rng=rng_f)
        rho_inf = rhos[int(0.8 * rhos.size):].mean()
        fwd.append(rho_inf)
        state = np.random.default_rng(seed=None).random(n) < rho_inf

    return np.asarray(lam_asc), np.asarray(fwd), np.asarray(back[::-1])
