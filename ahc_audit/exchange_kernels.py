"""Numba kernels of the same-group exchange search (RQ1, first bullet). Optional: needs numba.

Integer per-unit numerator Y_e = sum_{g,C} max(s n_{e,g,C} - n_{e,g} n_{e,C}, 0), so that
vol * HGP = sum_e Y_e / s_e (s_e = |e|). A step proposes k exchanges of the clusters of two nodes of the same
group in different clusters and applies the best one if it raises HGP.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def unit_Y(e, Hptr, Hidx, g, c):
    a = Hptr[e]; b = Hptr[e + 1]; s = b - a
    Y = 0
    for i in range(a, b):
        gi = g[Hidx[i]]; ci = c[Hidx[i]]
        first = True
        for j in range(a, i):
            if g[Hidx[j]] == gi and c[Hidx[j]] == ci:
                first = False
                break
        if not first:
            continue
        ng = 0; nc = 0; nl = 0
        for j in range(a, b):
            gj = g[Hidx[j]]; cj = c[Hidx[j]]
            if gj == gi:
                ng += 1
                if cj == ci:
                    nl += 1
            if cj == ci:
                nc += 1
        t = s * nl - ng * nc
        if t > 0:
            Y += t
    return Y


@njit(cache=True)
def all_Y(Hptr, Hidx, g, c):
    m = Hptr.size - 1
    Y = np.zeros(m, dtype=np.int64)
    for e in range(m):
        Y[e] = unit_Y(e, Hptr, Hidx, g, c)
    return Y


@njit(cache=True)
def seed_rng(seed):
    np.random.seed(seed)


@njit(cache=True)
def eval_swap(u, v, Hptr, Hidx, Tptr, Tidx, s, g, c, Y, mark, stamp):
    cu = c[u]; cv = c[v]
    c[u] = cv; c[v] = cu
    d = 0.0
    for t in range(Tptr[u], Tptr[u + 1]):
        e = Tidx[t]
        mark[e] = stamp
        d += (unit_Y(e, Hptr, Hidx, g, c) - Y[e]) / s[e]
    for t in range(Tptr[v], Tptr[v + 1]):
        e = Tidx[t]
        if mark[e] != stamp:
            d += (unit_Y(e, Hptr, Hidx, g, c) - Y[e]) / s[e]
    c[u] = cu; c[v] = cv
    return d


@njit(cache=True)
def apply_swap(u, v, Hptr, Hidx, Tptr, Tidx, g, c, Y, order, pos):
    cu = c[u]; cv = c[v]
    c[u] = cv; c[v] = cu
    for t in range(Tptr[u], Tptr[u + 1]):
        e = Tidx[t]
        Y[e] = unit_Y(e, Hptr, Hidx, g, c)
    for t in range(Tptr[v], Tptr[v + 1]):
        e = Tidx[t]
        Y[e] = unit_Y(e, Hptr, Hidx, g, c)
    pu = pos[u]; pv = pos[v]
    order[pu] = v; order[pv] = u
    pos[u] = pv; pos[v] = pu


@njit(cache=True)
def propose(elig, Hptr, Hidx, Tptr, Tidx, g, c, K, gptr, gidx, order, bstart, bsize):
    u = elig[np.random.randint(elig.size)]
    gu = g[u]; cu = c[u]
    C = -1
    du = Tptr[u + 1] - Tptr[u]
    if np.random.random() < 0.5 and du > 0:
        e = Tidx[Tptr[u] + np.random.randint(du)]
        w = Hidx[Hptr[e] + np.random.randint(Hptr[e + 1] - Hptr[e])]
        if c[w] != cu:
            C = c[w]
    if C < 0:
        ng = gptr[gu + 1] - gptr[gu]
        for _ in range(32):
            w = gidx[gptr[gu] + np.random.randint(ng)]
            if c[w] != cu:
                C = c[w]
                break
    if C < 0:
        return -1, -1
    blk = gu * K + C
    if bsize[blk] == 0:
        return -1, -1
    v = order[bstart[blk] + np.random.randint(bsize[blk])]
    return u, v


@njit(cache=True)
def run_steps(nsteps, k, elig, Hptr, Hidx, Tptr, Tidx, s, g, c, Y, mark, stamp0, K, gptr, gidx, order, pos,
              bstart, bsize, total, sw_u, sw_v, traj, idle, idle_max):
    """Run up to nsteps best-of-k steps. Returns (n_applied, total, idle, stamp, n_evals, n_steps)."""
    stamp = stamp0
    na = 0; nev = 0; nst = 0
    for _ in range(nsteps):
        if idle >= idle_max:
            break
        nst += 1
        best = 1e-12; bu = -1; bv = -1
        for _j in range(k):
            u, v = propose(elig, Hptr, Hidx, Tptr, Tidx, g, c, K, gptr, gidx, order, bstart, bsize)
            if u < 0:
                continue
            stamp += 1
            d = eval_swap(u, v, Hptr, Hidx, Tptr, Tidx, s, g, c, Y, mark, stamp)
            nev += 1
            if d > best:
                best = d; bu = u; bv = v
        if bu < 0:
            idle += 1
            continue
        apply_swap(bu, bv, Hptr, Hidx, Tptr, Tidx, g, c, Y, order, pos)
        total += best
        sw_u[na] = bu; sw_v[na] = bv; traj[na] = total
        na += 1
        idle = 0
    return na, total, idle, stamp, nev, nst
