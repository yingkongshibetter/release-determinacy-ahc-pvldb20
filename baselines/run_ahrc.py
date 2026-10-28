"""AHRC partition (Feng, Qiao, Piao, Cheng, PACMMOD 2025), local re-implementation.

This is the pipeline that produced the six audited AHRC partitions. It follows
the AHRC paper but is not the authors' code (see baselines/README.md):
    T   = row_norm(H^T) @ row_norm(H)                 hypergraph random walk
    ST  = alpha * sum_{l=0..gamma} (1-alpha)^l T^l      each T^l pruned to its
                                                      top_k_T largest entries per row
    SA  = M + M^T, M = cosine kNN graph of the node features (knn neighbours)
    S   = sqrt(row_norm(ST) @ row_norm(SA)), pruned to top_s entries per row
    partition = Louvain (igraph community_multilevel, resolution 1) on max(S, S^T)

AHRC returns its natural number of clusters; --k is only compared with the result.

Example:
    python baselines/run_ahrc.py --dataset openalex_db
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

import common


def row_normalize(M: sp.spmatrix, eps: float = 1e-12) -> sp.csr_matrix:
    M = M.tocsr().astype(np.float64)
    rs = np.asarray(M.sum(axis=1)).ravel()
    rs[rs < eps] = 1.0
    return (sp.diags(1.0 / rs) @ M).tocsr()


def top_k_per_row(M: sp.csr_matrix, k: int) -> sp.csr_matrix:
    M = M.tocsr()
    if k <= 0:
        return M
    rows, cols, vals = [], [], []
    for r in range(M.shape[0]):
        s, e = M.indptr[r], M.indptr[r + 1]
        if e - s <= k:
            rows.extend([r] * (e - s))
            cols.extend(M.indices[s:e].tolist())
            vals.extend(M.data[s:e].tolist())
        else:
            data = M.data[s:e]
            idx = M.indices[s:e]
            top = np.argpartition(data, -k)[-k:]
            rows.extend([r] * k)
            cols.extend(idx[top].tolist())
            vals.extend(data[top].tolist())
    return sp.csr_matrix((vals, (rows, cols)), shape=M.shape)


def build_SA(X: sp.csr_matrix, K: int) -> sp.csr_matrix:
    """SA = M + M^T for the cosine K-nearest-neighbour graph M (self excluded)."""
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=min(n, K + 10), metric="cosine", n_jobs=-1)
    nbrs.fit(X)
    dist, idx = nbrs.kneighbors(X, return_distance=True)
    sim = (1.0 - dist).astype(np.float64)
    rows, cols, vals = [], [], []
    for i in range(n):
        mask = idx[i] != i
        nb, sm = idx[i][mask], sim[i][mask]
        if nb.size >= K:
            nb, sm = nb[:K], sm[:K]
        rows.extend([i] * nb.size)
        cols.extend(nb.tolist())
        vals.extend(np.clip(sm, 0.0, 1.0).tolist())
    M = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    SA = (M + M.T).tocsr()
    SA.setdiag(0.0)
    SA.eliminate_zeros()
    return SA


def build_ST(H: sp.csr_matrix, alpha: float, gamma: int, top_k_T: int) -> sp.csr_matrix:
    Hf = H.astype(np.float64)
    T = (row_normalize(Hf.T) @ row_normalize(Hf)).tocsr()
    T = top_k_per_row(T, top_k_T)
    n = T.shape[0]
    ST = alpha * sp.eye(n, format="csr")
    Tl = sp.eye(n, format="csr")
    for l in range(1, gamma + 1):
        Tl = top_k_per_row((Tl @ T).tocsr(), top_k_T)
        ST = ST + (alpha * (1.0 - alpha) ** l) * Tl
    return ST.tocsr()


def build_S(ST: sp.csr_matrix, SA: sp.csr_matrix, top_s: int) -> sp.csr_matrix:
    P = (row_normalize(ST) @ row_normalize(SA)).tocsr()
    P.data = np.sqrt(np.clip(P.data, 0.0, None))
    return top_k_per_row(P, top_s)


def louvain_igraph(M: sp.csr_matrix, seed: int) -> np.ndarray:
    import igraph as ig
    M_sym = M.tocsr().maximum(M.tocsr().T)
    M_sym.setdiag(0.0)
    M_sym.eliminate_zeros()
    if M_sym.nnz == 0:
        return np.zeros(M_sym.shape[0], dtype=np.int64)
    coo = M_sym.tocoo()
    mask = coo.row < coo.col
    g = ig.Graph(n=int(M_sym.shape[0]),
                 edges=list(zip(coo.row[mask].tolist(), coo.col[mask].tolist())),
                 directed=False)
    g.es["weight"] = coo.data[mask].astype(float).tolist()
    ig.set_random_number_generator(random.Random(int(seed)))
    membership = g.community_multilevel(weights="weight", resolution=1.0)
    return np.asarray(membership.membership, dtype=np.int64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p, "ahrc")
    p.add_argument("--knn", type=int, default=20, help="neighbours of the attribute kNN graph")
    p.add_argument("--top-s", type=int, default=100, help="entries kept per row of S")
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--gamma", type=int, default=2)
    p.add_argument("--top-k-t", type=int, default=200, help="entries kept per row of each T^l")
    a = common.resolve_common_args(p.parse_args(), "ahrc")
    try:
        import igraph  # noqa: F401
    except ImportError:
        raise SystemExit("run_ahrc.py needs python-igraph (the audited partitions used igraph 1.0.0)")

    t0 = time.time()
    H = common.load_incidence(a.data_dir)
    X = common.load_features(a.data_dir)
    if X.shape[0] != H.shape[1]:
        raise SystemExit(f"features have {X.shape[0]} rows, incidence has {H.shape[1]} nodes")
    ST = build_ST(H, a.alpha, a.gamma, a.top_k_t)
    SA = build_SA(X, a.knn)
    S = build_S(ST, SA, a.top_s)
    clusters = louvain_igraph(S, a.seed)
    common.write_partition(a.out, clusters)
    common.report("ahrc", a.dataset, clusters, a.k, a.out)
    print(f"[ahrc] {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
