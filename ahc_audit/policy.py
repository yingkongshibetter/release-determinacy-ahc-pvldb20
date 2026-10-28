"""Cluster-level invitation policies and their decision gaps (Proposition 1; RQ1, third bullet).

A policy pi: clusters -> [0, 1] invites everyone in cluster C with probability pi(C). For a unit e and a
group g present in e, Phi_g(e; pi) = sum_C pi(C) q_{e,g}(C) and Phi(e; pi) = sum_C pi(C) q_e(C); the
membership-weighted average gap
    G(pi) = (1/vol) sum_{e,g} n_{e,g} |Phi_g(e; pi) - Phi(e; pi)|
is at most HGP for every pi (Prop. 1(b)). G is convex in pi, so its maximum is attained at a 0/1 policy.
worst_policy() starts from the best of 400 random 0/1 policies (seeds SeedSequence(20261025, (dataset index,
method index)), the same draws for every attribute of a partition), then flips single clusters while G
increases (at most 30 sweeps), and enumerates all 2^K policies when K <= 12.
"""
from __future__ import annotations

import numpy as np

from .data import compact
from .hgp import onehot

R_DRAWS = 400
CHUNK = 25
SEED = 20261025


def cell_arrays(H, cl, g):
    cl, k = compact(cl)
    g, G = compact(g)
    s = np.diff(H.indptr).astype(np.int64)
    Mp = (H @ onehot(g * k + cl, G * k)).tocoo()
    e = Mp.row.astype(np.int64)
    gg = (Mp.col // k).astype(np.int64)
    cc = (Mp.col % k).astype(np.int64)
    cnt = Mp.data.astype(np.float64)
    uk, eg = np.unique(e * G + gg, return_inverse=True)
    eg = eg.ravel()
    return dict(e=e, cc=cc, cnt=cnt, eg=eg, n_eg=np.bincount(eg, weights=cnt), e_of_eg=uk // G,
                s=s.astype(np.float64), k=k, vol=float(s.sum()), m=H.shape[0])


def gap_draws(A, PI):
    """G(pi) for each row of a (r, K) 0/1 policy matrix."""
    out = np.empty(PI.shape[0])
    for i in range(PI.shape[0]):
        w = PI[i][A["cc"]] * A["cnt"]
        a = np.bincount(A["eg"], weights=w, minlength=A["n_eg"].size)
        b = np.bincount(A["e"], weights=w, minlength=A["m"]) / A["s"]
        out[i] = np.abs(a - A["n_eg"] * b[A["e_of_eg"]]).sum() / A["vol"]
    return out


def greedy(A, pi0, max_sweeps=30):
    e, cc, cnt, eg, n_eg, e_of_eg, s = A["e"], A["cc"], A["cnt"], A["eg"], A["n_eg"], A["e_of_eg"], A["s"]
    k = A["k"]
    pi = pi0.copy()
    w = pi[cc] * cnt
    a = np.bincount(eg, weights=w, minlength=n_eg.size)
    bs = np.bincount(e, weights=w, minlength=A["m"])
    order = np.argsort(cc, kind="stable")
    cuts = np.searchsorted(cc[order], np.arange(k + 1))
    eg_cuts = np.searchsorted(e_of_eg, np.arange(A["m"] + 1))
    G = np.abs(a - n_eg * bs[e_of_eg] / s[e_of_eg]).sum()
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        improved = False
        for C in range(k):
            ent = order[cuts[C]:cuts[C + 1]]
            if ent.size == 0:
                continue
            sign = -1.0 if pi[C] == 1 else 1.0
            units = np.unique(e[ent])
            aff = np.concatenate([np.arange(eg_cuts[u], eg_cuts[u + 1]) for u in units])
            old = np.abs(a[aff] - n_eg[aff] * bs[e_of_eg[aff]] / s[e_of_eg[aff]]).sum()
            da = np.bincount(np.searchsorted(aff, eg[ent]), weights=sign * cnt[ent], minlength=aff.size)
            dbu = np.bincount(np.searchsorted(units, e[ent]), weights=sign * cnt[ent], minlength=units.size)
            ua = np.searchsorted(units, e_of_eg[aff])
            new = np.abs(a[aff] + da - n_eg[aff] * (bs[e_of_eg[aff]] + dbu[ua]) / s[e_of_eg[aff]]).sum()
            if new > old + 1e-12:
                pi[C] = 1 - pi[C]
                a[aff] += da
                bs[units] += dbu
                G += new - old
                improved = True
        if not improved:
            break
    Gx = np.abs(a - n_eg * bs[e_of_eg] / s[e_of_eg]).sum() / A["vol"]
    assert abs(Gx - G / A["vol"]) < 1e-9
    return Gx, pi, sweeps


def draws(k, dataset_index, method_index):
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(dataset_index, method_index)))
    return (rng.random((R_DRAWS, k)) < 0.5).astype(np.float64)


def worst_policy(H, cl, g, PI):
    """(largest G over the random draws, greedy G from the best draw, exact max over 2^K policies or nan)."""
    A = cell_arrays(H, cl, g)
    Gs = np.concatenate([gap_draws(A, PI[j:j + CHUNK]) for j in range(0, PI.shape[0], CHUNK)])
    b = int(np.argmax(Gs))
    Gg, _, _ = greedy(A, PI[b])
    Gex = np.nan
    k = A["k"]
    if k <= 12:
        allpi = ((np.arange(2 ** k)[:, None] >> np.arange(k)[None, :]) & 1).astype(np.float64)
        Gex = float(max(gap_draws(A, allpi[j:j + 256]).max() for j in range(0, 2 ** k, 256)))
    return float(Gs[b]), float(Gg), Gex, float(Gs.mean())
