"""Releases computed by the producer from the incidence (Section 2.1).

    unweighted clique     every pair contained in at least one unit, without unit identifiers
    size-only weighted    w_uv = sum_{e containing u,v} omega(|e|); the normalized clique has omega(k) = 1/(k-1)
    random-walk release   P = M diag(|e|)^-1 M^T (diagonal included), or T = diag(d)^-1 P with the degrees d
All matrices here are built in float64 for scanning; exact values follow from the incidence (hgp.py).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import scipy.sparse as sp


def clique_pair_keys(H: sp.csr_matrix) -> np.ndarray:
    """Sorted unique keys u * n + v (u < v) of the unweighted clique."""
    H = H.tocsr()
    n = H.shape[1]
    ind, idx = H.indptr, H.indices
    s = np.diff(ind)
    chunks = []
    for sz in np.unique(s):
        if sz < 2:
            continue
        rows = np.where(s == sz)[0]
        mem = np.sort(np.stack([idx[ind[r]:ind[r + 1]] for r in rows]), axis=1).astype(np.int64)
        iu, iv = np.triu_indices(int(sz), 1)
        chunks.append(np.unique(mem[:, iu] * n + mem[:, iv]))
    return np.unique(np.concatenate(chunks)) if chunks else np.zeros(0, dtype=np.int64)


def unweighted_clique(H):
    """Pair lists (u, v), u < v, of the unweighted clique."""
    n = H.shape[1]
    keys = clique_pair_keys(H)
    return keys // n, keys % n


def weighted_clique(H, omega) -> sp.coo_matrix:
    """Size-only weighted clique, upper triangle without self-loops; omega maps a unit size to a weight."""
    H = H.tocsr().astype(np.float64)
    s = np.diff(H.indptr)
    w = np.array([omega(int(k)) if k > 1 else 0.0 for k in s], dtype=np.float64)
    W = (H.T @ sp.diags(w) @ H).tocsr()
    return sp.triu(W, k=1).tocoo()


def normalized_clique(H) -> sp.coo_matrix:
    """omega(k) = 1/(k-1), the weighting of the public cond-mat coauthorship network."""
    return weighted_clique(H, lambda k: 1.0 / (k - 1))


def clique_expansion(H) -> sp.coo_matrix:
    """omega(k) = 1/C(k,2), the clique expansion that CE+Fair-Louvain clusters."""
    return weighted_clique(H, lambda k: 2.0 / (k * (k - 1)))


def random_walk_P(H) -> sp.coo_matrix:
    """P_uv = sum_{e containing u,v} 1/|e|, including the diagonal (symmetric, full)."""
    Hf = H.astype(np.float64)
    s = np.diff(H.indptr).astype(np.float64)
    P = (Hf.T @ sp.diags(1.0 / s) @ Hf).tocoo()
    P.sum_duplicates()
    return P


def degrees(H) -> np.ndarray:
    return np.asarray(H.sum(axis=0)).ravel().astype(np.int64)


def row_normalize(M, eps=1e-12):
    M = M.tocsr().astype(np.float64)
    rs = np.asarray(M.sum(axis=1)).ravel()
    rs[rs < eps] = 1.0
    return (sp.diags(1.0 / rs) @ M).tocsr()


def transition_T(H) -> sp.csr_matrix:
    """T = diag(d)^-1 P in float64, computed as in the AHRC and AHCKA code (row-normalized H^T times H)."""
    Hf = H.astype(np.float64)
    T = (row_normalize(Hf.T) @ row_normalize(Hf)).tocsr()
    T.sort_indices()
    return T


def top_k_per_row(M, k: int) -> sp.csr_matrix:
    """Keep the k largest entries of each row without rescaling, as AHRC prunes its operator."""
    M = M.tocsr()
    if k <= 0:
        return M
    n = M.shape[0]
    rows, cols, vals = [], [], []
    for r in range(n):
        a, b = M.indptr[r], M.indptr[r + 1]
        if b - a <= k:
            rows.extend([r] * (b - a)); cols.extend(M.indices[a:b].tolist()); vals.extend(M.data[a:b].tolist())
        else:
            data = M.data[a:b]; idx = M.indices[a:b]
            top = np.argpartition(data, -k)[-k:]
            rows.extend([r] * k); cols.extend(idx[top].tolist()); vals.extend(data[top].tolist())
    return sp.csr_matrix((vals, (rows, cols)), shape=M.shape)


def check_P(P: sp.coo_matrix, d: np.ndarray, tol=1e-9) -> bool:
    """The auditor's one-pass check of a received P: symmetric, nonnegative, row sums equal to the degrees."""
    Pc = P.tocsr()
    sym = abs(Pc - Pc.T).max() <= tol if Pc.nnz else True
    nonneg = bool((Pc.data >= 0).all())
    rows = np.asarray(Pc.sum(axis=1)).ravel()
    return bool(sym and nonneg and np.allclose(rows, d, atol=tol, rtol=0))


class ExactEntries:
    """Exact release entries from the incidence, for checks and for ambiguous roundings."""

    def __init__(self, H):
        Hc = H.tocsc()
        self.ind, self.ptr = Hc.indices, Hc.indptr
        self.s = np.diff(H.tocsr().indptr)
        self.d = np.diff(Hc.indptr)

    def common(self, u, v):
        a = self.ind[self.ptr[u]:self.ptr[u + 1]]
        if u == v:
            return a
        b = self.ind[self.ptr[v]:self.ptr[v + 1]]
        return np.intersect1d(a, b, assume_unique=True)

    def w(self, u, v):
        """Normalized-clique weight."""
        return sum(Fraction(1, int(self.s[e]) - 1) for e in self.common(u, v))

    def p(self, u, v):
        """Entry of P."""
        return sum(Fraction(1, int(self.s[e])) for e in self.common(u, v))

    def t(self, u, v):
        """Entry of T."""
        return self.p(u, v) / int(self.d[u])
