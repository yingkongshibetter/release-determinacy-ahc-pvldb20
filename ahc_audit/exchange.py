"""Same-group exchanges (Remark 1; RQ1, first bullet).

Exchanging the clusters of two nodes of the same group keeps every cluster size and every count n_{g,C},
so every audit of cluster-level counts (CGP, balance, the parity gap of each cluster) keeps its value.
A hill-climbing search over such exchanges raises HGP (search(), needs numba); a witness is the list of
exchanges, and verify() replays it in exact arithmetic without numba.

Search settings: best of K_PROPOSALS = 64 proposals per step, stop after IDLE_MAX = 5000 steps without an
improving exchange or at the time cap (300 s per cell in the paper), seed SeedSequence(20261024, (cell index,)).
"""
from __future__ import annotations

import time
from fractions import Fraction

import numpy as np

K_PROPOSALS = 64
IDLE_MAX = 5000
CHUNK = 100
SEED_ROOT = 20261024


def exact_hgp_csr(Hptr, Hidx, s, g, c, G, K) -> Fraction:
    e = np.repeat(np.arange(s.size, dtype=np.int64), s)
    gg = g[Hidx]; cc = c[Hidx]
    k3, n3 = np.unique((e * G + gg) * K + cc, return_counts=True)
    e3 = k3 // (G * K); g3 = (k3 // K) % G; c3 = k3 % K
    kg, ng = np.unique(e * G + gg, return_counts=True)
    kc, nc = np.unique(e * K + cc, return_counts=True)
    ng3 = ng[np.searchsorted(kg, e3 * G + g3)]
    nc3 = nc[np.searchsorted(kc, e3 * K + c3)]
    t = s[e3] * n3.astype(np.int64) - ng3.astype(np.int64) * nc3.astype(np.int64)
    Y = np.zeros(s.size, dtype=np.int64)
    np.add.at(Y, e3, np.maximum(t, 0))
    num = Fraction(0)
    for kk in np.unique(s):
        num += Fraction(int(Y[s == kk].sum()), int(kk))
    return num / int(s.sum())


def cluster_audits(g, c, G, K):
    """Counts n_{g,C}, CGP, balance and the largest parity gap of a cluster, exactly."""
    n = g.size
    T = np.bincount(g * K + c, minlength=G * K).reshape(G, K).astype(object)
    sz = T.sum(0); ng = T.sum(1)
    cgp = Fraction(sum(abs(n * T[a, b] - sz[b] * ng[a]) for a in range(G) for b in range(K) if sz[b] > 0), 2 * n * n)
    bal = None; gap = Fraction(0)
    for b in range(K):
        if sz[b] == 0:
            continue
        for a in range(G):
            r = Fraction(int(T[a, b]) * n, int(sz[b]) * int(ng[a]))
            v = min(r, 1 / r) if r > 0 else Fraction(0)
            bal = v if bal is None else min(bal, v)
            gap = max(gap, abs(Fraction(int(T[a, b]), int(sz[b])) - Fraction(int(ng[a]), n)))
    return np.asarray(T, dtype=np.int64), cgp, bal, gap


def _arrays(H, g, cl):
    from .data import compact
    c, K = compact(cl)
    gg, G = compact(g)
    Hptr = H.indptr.astype(np.int64); Hidx = H.indices.astype(np.int64)
    s = np.diff(Hptr).astype(np.int64)
    return Hptr, Hidx, s, gg, c, G, K


def search(H, g, cl, seed, tcap=300.0, log=None):
    """Run the hill-climbing search; returns (swap_u, swap_v, traj, info)."""
    from . import exchange_kernels as CK
    m, n = H.shape
    Hptr, Hidx, s, g, c, G, K = _arrays(H, g, cl)
    c = c.copy()
    T = H.T.tocsr(); T.sort_indices()
    Tptr = T.indptr.astype(np.int64); Tidx = T.indices.astype(np.int64)
    vol = int(s.sum())
    key = g * K + c
    order = np.argsort(key, kind="stable").astype(np.int64)
    pos = np.empty(n, dtype=np.int64); pos[order] = np.arange(n)
    bsize = np.bincount(key, minlength=G * K).astype(np.int64)
    bstart = np.concatenate([[0], np.cumsum(bsize)[:-1]]).astype(np.int64)
    gord = np.argsort(g, kind="stable").astype(np.int64)
    gptr = np.concatenate([[0], np.cumsum(np.bincount(g, minlength=G))]).astype(np.int64)
    ncl_g = (bsize.reshape(G, K) > 0).sum(1)
    elig = np.where(ncl_g[g] >= 2)[0].astype(np.int64)
    Y = CK.all_Y(Hptr, Hidx, g, c)
    total = float((Y / s).sum())
    CK.seed_rng(int(seed))
    mark = np.zeros(m, dtype=np.int64)
    su, sv, tr = [], [], []
    bu = np.empty(CHUNK, dtype=np.int64); bv = np.empty(CHUNK, dtype=np.int64); bt = np.empty(CHUNK)
    stamp = 0; idle = 0; steps = 0; evals = 0
    t0 = time.perf_counter(); stop = "time"
    while True:
        if elig.size == 0:
            stop = "no_eligible"; break
        na, total_new, idle, stamp, nev, nst = CK.run_steps(CHUNK, K_PROPOSALS, elig, Hptr, Hidx, Tptr, Tidx, s, g, c,
                                                            Y, mark, stamp, K, gptr, gord, order, pos, bstart, bsize,
                                                            total, bu, bv, bt, idle, IDLE_MAX)
        evals += nev; steps += nst
        if na:
            su.append(bu[:na].copy()); sv.append(bv[:na].copy()); tr.append(bt[:na] / vol)
        total = float((Y / s).sum())
        if idle >= IDLE_MAX:
            stop = "idle"; break
        if time.perf_counter() - t0 >= tcap:
            stop = "time"; break
    cat = lambda x, dt: np.concatenate(x) if x else np.zeros(0, dtype=dt)
    info = dict(swaps=int(sum(len(x) for x in su)), steps=steps, evals=evals, stop=stop,
                t_search_s=time.perf_counter() - t0, seed=int(seed))
    return cat(su, np.int64).astype(np.int32), cat(sv, np.int64).astype(np.int32), cat(tr, np.float64), info


def first_crossing(H, g, cl, su, sv, traj, threshold: Fraction):
    """Smallest prefix length t with exact HGP > threshold (located on the float trajectory, checked exactly)."""
    Hptr, Hidx, s, gg, c0, G, K = _arrays(H, g, cl)
    over = np.flatnonzero(traj > float(threshold))
    if over.size == 0:
        return None

    def h_at(t):
        c = c0.copy()
        for u, v in zip(su[:t], sv[:t]):
            c[u], c[v] = c[v], c[u]
        return exact_hgp_csr(Hptr, Hidx, s, gg, c, G, K)
    t = int(over[0]) + 1
    while t > 1 and h_at(t - 1) > threshold:      # float trajectory crossed late: step back
        t -= 1
    while h_at(t) <= threshold:                   # float trajectory crossed early: step forward
        t += 1
    return t


def verify(H, g, cl, su, sv, traj, B: Fraction, tol=Fraction(1, 10)):
    """Replay a witness exactly: every exchange is between two nodes of one group in different clusters, every
    cluster-level audit is unchanged, and the final HGP; also the first crossings of B and of 0.1 and the
    number of distinct nodes whose cluster differs from the natural partition at the crossing of B."""
    Hptr, Hidx, s, gg, c0, G, K = _arrays(H, g, cl)
    n = c0.size
    T0, cgp0, bal0, gap0 = cluster_audits(gg, c0, G, K)
    h0 = exact_hgp_csr(Hptr, Hidx, s, gg, c0, G, K)
    c = c0.copy(); ok = True
    for u, v in zip(su.astype(np.int64), sv.astype(np.int64)):
        ok &= bool(gg[u] == gg[v] and c[u] != c[v])
        c[u], c[v] = c[v], c[u]
    T1, cgp1, bal1, gap1 = cluster_audits(gg, c, G, K)
    h1 = exact_hgp_csr(Hptr, Hidx, s, gg, c, G, K)
    out = dict(HGP=float(h0), HGP_end=float(h1), delta=float(h1 - h0), swaps=int(su.size),
               moves_same_group_cross_cluster=ok,
               cluster_audits_identical=bool(np.array_equal(T0, T1) and cgp0 == cgp1 and bal0 == bal1 and gap0 == gap1),
               final_matches_trajectory=bool(abs(float(h1) - (float(traj[-1]) if traj.size else float(h0))) < 1e-9),
               pass_B=bool(h0 <= B), fails_B_reached=bool(h1 > B), pass_01=bool(h0 <= tol), fails_01_reached=bool(h1 > tol))
    for name, thr, passes in (("B", B, h0 <= B), ("01", tol, h0 <= tol)):
        t = first_crossing(H, g, cl, su, sv, traj, thr) if passes else None
        out[f"t_{name}"] = t
        if t is not None:
            cm = c0.copy()
            for u, v in zip(su[:t].astype(np.int64), sv[:t].astype(np.int64)):
                cm[u], cm[v] = cm[v], cm[u]
            out[f"moved_share_{name}"] = int((cm != c0).sum()) / n
    return out
