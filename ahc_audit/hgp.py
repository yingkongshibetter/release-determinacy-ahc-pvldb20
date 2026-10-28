"""Exact HGP, CGP, interaction index and dissimilarity from incidence.

For a unit e with |e| = s members, n_{e,g,C} members of group g in cluster C, and the marginals
n_{e,g} and n_{e,C}, the integers
    Y_e = sum_{g,C} max(s n_{e,g,C} - n_{e,g} n_{e,C}, 0)   = s^2 psi_e          (Eq. (4))
    X_e = ordered member pairs that differ in group and cluster = 2 N_e         (Lemma 1)
    Z_e = ordered member pairs that differ in group = s^2 - sum_g n_{e,g}^2     = s^2 I_e
give, with vol = sum_e s_e,
    vol HGP = sum_e Y_e / s_e,   vol U_P = sum_e X_e / s_e,   vol U_Q = sum_e X_e / (s_e - 1),
    vol Ibar = sum_e Z_e / s_e,  Dbar = HGP / Ibar.
Every sum is an integer per unit size, so each statistic is an exact rational (Fraction).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import compact


def onehot(lab: np.ndarray, k: int) -> sp.csr_matrix:
    n = lab.size
    return sp.csr_matrix((np.ones(n, dtype=np.int64), (np.arange(n), lab)), shape=(n, k))


def unit_counts(H: sp.csr_matrix, cl: np.ndarray, g: np.ndarray):
    """Per-unit integers (s, X, Y, Z) and the number of nonzero contingency rows n_{e,g,C}.

    H is a 0/1 int64 CSR matrix (units x nodes)."""
    cl, k = compact(cl)
    g, G = compact(g)
    s = np.diff(H.indptr).astype(np.int64)
    Mg = (H @ onehot(g, G)).tocsr()
    Mc = (H @ onehot(cl, k)).tocsr()
    Mp = (H @ onehot(g * k + cl, G * k)).tocoo()

    def sq(M):
        return np.asarray(M.multiply(M).sum(axis=1)).ravel().astype(np.int64)

    sqg = sq(Mg)
    X = s * s - sqg - sq(Mc) + sq(Mp.tocsr())
    Z = s * s - sqg
    e, cell, obs = Mp.row, Mp.col, Mp.data.astype(np.int64)
    gg, cc = cell // k, cell % k
    meg = np.asarray(Mg[e, gg]).ravel().astype(np.int64)
    mec = np.asarray(Mc[e, cc]).ravel().astype(np.int64)
    pos = np.maximum(s[e] * obs - meg * mec, 0)
    # integer sums in float64 are exact here (each term is at most |e|^2)
    Y = np.bincount(e, weights=pos.astype(np.float64), minlength=H.shape[0]).round().astype(np.int64)
    return s, X, Y, Z, int(Mp.nnz)


def contingency_rows(H, cl, g):
    """Rows (e, n_{e,g,C}, n_{e,g}, n_{e,C}) of the contingency table, with unit sizes s."""
    cl, k = compact(cl)
    g, G = compact(g)
    s = np.diff(H.indptr).astype(np.int64)
    Mg = (H @ onehot(g, G)).tocsr()
    Mc = (H @ onehot(cl, k)).tocsr()
    Mp = (H @ onehot(g * k + cl, G * k)).tocoo()
    e, cell, obs = Mp.row.astype(np.int64), Mp.col, Mp.data.astype(np.int64)
    gg, cc = cell // k, cell % k
    meg = np.asarray(Mg[e, gg]).ravel().astype(np.int64)
    mec = np.asarray(Mc[e, cc]).ravel().astype(np.int64)
    return s, e, obs, meg, mec, gg.astype(np.int64), cc.astype(np.int64)


def size_sums(s: np.ndarray, v: np.ndarray, shift: int = 0) -> dict:
    """{unit size k: python-int sum of v over units of size k} for sizes with k - shift > 0."""
    out = {}
    for k in np.unique(s):
        if int(k) - shift <= 0:
            continue
        out[int(k)] = int(v[s == k].sum())
    return out


def exact_sum(s: np.ndarray, v: np.ndarray, shift: int = 0) -> Fraction:
    """sum_e v_e / (s_e - shift) as an exact Fraction."""
    return sum((Fraction(t, k - shift) for k, t in size_sums(s, v, shift).items()), Fraction(0))


def per_size_counts_and_sums(H, cl, g) -> list[tuple[int, int, int]]:
    """(unit size k, number of units of size k, sum of Y_e = |e|^2 psi_e over them): what the SQL query returns."""
    s, _, Y, _, _ = unit_counts(H, cl, g)
    return [(int(k), int((s == k).sum()), int(Y[s == k].sum())) for k in np.unique(s)]


def hgp_from_size_sums(rows) -> Fraction:
    """Exact HGP from per-size (k, units, sum Y) rows: two integers per unit size."""
    vol = sum(k * u for k, u, _ in rows)
    return sum((Fraction(y, k) for k, _, y in rows), Fraction(0)) / vol


def hgp_exact(H, cl, g) -> Fraction:
    s, _, Y, _, _ = unit_counts(H, cl, g)
    return exact_sum(s, Y) / int(s.sum())


def hgp_float(H, cl, g) -> float:
    s, _, Y, _, _ = unit_counts(H, cl, g)
    return float((Y / s).sum() / s.sum())


def cgp_exact(cl, g) -> Fraction:
    """CGP = sum_C |C|/n TV(p_{.|C}, p) (Eq. (1)), exactly: sum |n n_{gC} - n_C n_g| / (2 n^2)."""
    c, k = compact(cl)
    gg, G = compact(g)
    n = c.size
    cnt = np.bincount(c * G + gg, minlength=k * G).reshape(k, G).astype(np.int64)
    num = int(np.abs(n * cnt - np.outer(cnt.sum(1), cnt.sum(0))).sum())
    return Fraction(num, 2 * n * n)


def population_interaction(g) -> Fraction:
    """1 - sum_g p_g^2, the denominator of D_CGP = CGP / (1 - sum_g p_g^2)."""
    gg, _ = compact(g)
    pg = np.bincount(gg).astype(np.int64)
    return 1 - Fraction(int((pg * pg).sum()), int(pg.sum()) ** 2)


def mixed_membership_share(H, g) -> float:
    """Share of memberships that lie in units with two or more groups (the 30% screen of Section 5.1)."""
    gg, G = compact(g)
    Mg = (H @ onehot(gg, G)).tocsr()
    groups_per_unit = np.diff(Mg.indptr)
    s = np.diff(H.indptr)
    return float(s[groups_per_unit >= 2].sum() / s.sum())


def split_membership_share(H, cl) -> float:
    """Share of memberships that lie in units split over two or more clusters (RQ1, last bullet)."""
    c, k = compact(cl)
    Mc = (H @ onehot(c, k)).tocsr()
    s = np.diff(H.indptr)
    return float(s[np.diff(Mc.indptr) >= 2].sum() / s.sum())


def mean_worst_gap(H, cl, g) -> float:
    """Membership-weighted average of M_e, the largest TV distance between the cluster distributions
    of two groups present in e (Section 2.3); units with one group have M_e = 0."""
    s, e, n, ng, _, gg, cc = contingency_rows(H, cl, g)
    df = pd.DataFrame({"e": e, "g": gg, "c": cc, "q": n / ng})
    groups = df[["e", "g"]].drop_duplicates()
    pairs = groups.merge(groups, on="e")
    pairs = pairs[pairs.g_x < pairs.g_y]
    if pairs.empty:
        return 0.0
    a = df.rename(columns={"g": "g_x", "q": "q_x"})
    b = df.rename(columns={"g": "g_y", "q": "q_y"})
    ov = a.merge(b, on=["e", "c"])
    ov = ov[ov.g_x < ov.g_y]
    ov["m"] = np.minimum(ov.q_x, ov.q_y)
    overlap = ov.groupby(["e", "g_x", "g_y"]).m.sum()
    pairs = pairs.set_index(["e", "g_x", "g_y"])
    pairs["tv"] = 1.0 - overlap.reindex(pairs.index).fillna(0.0)
    M = pairs.groupby(level=0).tv.max()
    Me = np.zeros(H.shape[0])
    Me[M.index.to_numpy()] = M.to_numpy()
    return float((s * Me).sum() / s.sum())


class CellStats:
    """Exact per-cell sums shared by HGP, U_P, U_Q, Ibar and Dbar."""

    def __init__(self, H, cl, g):
        s, X, Y, Z, rows = unit_counts(H, cl, g)
        self.vol = int(s.sum())
        self.rows = rows
        self.SY = exact_sum(s, Y)        # vol HGP
        self.SX = exact_sum(s, X)        # vol U_P
        self.SZ = exact_sum(s, Z)        # vol Ibar
        self.SX1 = exact_sum(s, X, 1)    # vol U_Q
        self.SZ1 = exact_sum(s, Z, 1)    # 2 vol Ibar of the normalized-clique pseudo-hypergraph
        self.max_size = int(s.max())
        self.min_size = int(s.min())

    @property
    def hgp(self) -> Fraction:
        return self.SY / self.vol

    @property
    def ibar(self) -> Fraction:
        return self.SZ / self.vol

    @property
    def dbar(self) -> Fraction:
        return self.SY / self.SZ if self.SZ else Fraction(0)
