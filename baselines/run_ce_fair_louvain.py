"""CE+Fair-Louvain partition: fairness-penalized local moves on the clique expansion.

Graph: the clique expansion of the units with the size-only weight
omega(k) = 1 / C(k, 2) per pair of a unit of size k (pairs summed over units,
no self loops). Objective, maximized by local moves:
    F(C) = Q(C) - lambda * sum_j w_j * sum_c (|c| / n) * max(0, TV_c(A_j) - eps_j)
where Q is weighted modularity on the clique expansion and TV_c(A_j) is the
total-variation distance between the distribution of attribute A_j in cluster
c and in the whole dataset. Settings of the audited partitions: lambda 1.0,
eps_j 0.10 and w_j 1.0 for every audited attribute of the dataset, singleton
start, at most 5 passes over the nodes in a seeded random order, a node moves
to the neighbouring cluster with the largest gain when the gain exceeds 1e-12.

There is no aggregation phase: the result is the partition after the local
moves, so the number of clusters is large (Table 3). This is a local
implementation, not code released with the Fair-Louvain paper (see README).

Example:
    python baselines/run_ce_fair_louvain.py --dataset openalex_db
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations

import numpy as np
import scipy.sparse as sp

import common


def build_clique_expansion(units: list[list[int]], n: int) -> sp.csr_matrix:
    """Weighted clique expansion; each pair of a unit of size k adds 1 / C(k, 2)."""
    rows, cols, vals = [], [], []
    for e in units:
        k = len(e)
        if k < 2:
            continue
        w = 1.0 / (k * (k - 1) / 2.0)
        for u, v in combinations(sorted(int(x) for x in e), 2):
            rows += [u, v]
            cols += [v, u]
            vals += [w, w]
    if not rows:
        return sp.csr_matrix((n, n), dtype=np.float64)
    M = sp.coo_matrix((np.asarray(vals, dtype=np.float64),
                       (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
                      shape=(n, n)).tocsr()
    M.sum_duplicates()
    return M


def fair_louvain_local_moves(G: sp.csr_matrix, init: np.ndarray, attrs: list[dict], *,
                             lambda_v: float, max_passes: int, seed: int) -> np.ndarray:
    """Local moves maximizing modularity minus the weighted excess-TV penalty.

    attrs: list of dicts with keys codes (int array, length n), num_groups,
    epsilon, weight. Returns contiguous cluster ids.
    """
    rng = np.random.default_rng(seed)
    n = G.shape[0]
    indptr, indices, data = G.indptr, G.indices, G.data
    inv2n = 1.0 / (2.0 * n)
    clusters = np.asarray(init, dtype=np.int64).copy()

    k_node = np.asarray(G.sum(axis=1)).ravel().astype(np.float64)
    total_2m = float(G.sum())
    if total_2m <= 0:
        raise ValueError("total edge weight is zero")
    inv2m = 1.0 / total_2m
    inv2m_sq_2 = 1.0 / (2.0 * total_2m * total_2m)
    K_max = int(clusters.max()) + 1
    sigma_tot = np.zeros(K_max, dtype=np.float64)
    np.add.at(sigma_tot, clusters, k_node)

    size = np.zeros(K_max, dtype=np.int64)
    for i in range(n):
        size[clusters[i]] += 1
    codes, cnt, phi, p_glob = [], [], [], []
    for spec in attrs:
        c = spec["codes"]
        G_ = int(spec["num_groups"])
        pg = np.bincount(c, minlength=G_).astype(float)
        pg /= max(pg.sum(), 1)
        ct = np.zeros((K_max, G_), dtype=np.int64)
        for i in range(n):
            ct[clusters[i], c[i]] += 1
        codes.append(c)
        cnt.append(ct)
        phi.append(ct.astype(np.float64) - size[:, None].astype(np.float64) * pg[None, :])
        p_glob.append(pg)
    eps_size = np.array([float(s["epsilon"]) for s in attrs]) / n
    weight = np.array([float(s["weight"]) for s in attrs], dtype=np.float64)

    for _ in range(max_passes):
        order = np.arange(n)
        rng.shuffle(order)
        moves = 0
        for i in order:
            ki = float(k_node[i])
            a = int(clusters[i])
            s_, e_ = int(indptr[i]), int(indptr[i + 1])
            if e_ == s_:
                continue
            nbr_clusters = clusters[indices[s_:e_]]
            ws = data[s_:e_].astype(np.float64)
            cand = np.unique(nbr_clusters)
            if a not in cand:
                cand = np.concatenate(([a], cand))
            k_ic = np.zeros(len(cand))
            cand_index = {c: idx for idx, c in enumerate(cand.tolist())}
            for jj in range(nbr_clusters.shape[0]):
                k_ic[cand_index[int(nbr_clusters[jj])]] += ws[jj]
            a_pos = cand_index[a]
            k_ia = float(k_ic[a_pos])
            delta_Q = (k_ic - k_ia) * inv2m + ki * (float(sigma_tot[a]) - sigma_tot[cand] - ki) * inv2m_sq_2

            size_a_old = int(size[a])
            size_a_new = size_a_old - 1
            size_c_old = size[cand].astype(np.float64)
            size_c_new = size_c_old + 1.0
            delta_U = np.zeros(len(cand), dtype=np.float64)
            for j in range(len(attrs)):
                g = int(codes[j][i])
                ph, pg, es = phi[j], p_glob[j], float(eps_size[j])
                pa_old = ph[a]
                pa_old_sum = float(np.abs(pa_old).sum())
                pa_new = pa_old + pg
                pa_new[g] -= 1.0
                pa_new_sum = float(np.abs(pa_new).sum())
                pc_old = ph[cand]
                pc_old_sum = np.abs(pc_old).sum(axis=1)
                pc_new = pc_old - pg[None, :]
                pc_new[:, g] += 1.0
                pc_new_sum = np.abs(pc_new).sum(axis=1)
                ta_old = max(0.0, pa_old_sum * inv2n - es * size_a_old)
                ta_new = max(0.0, pa_new_sum * inv2n - es * size_a_new)
                tc_old = np.maximum(0.0, pc_old_sum * inv2n - es * size_c_old)
                tc_new = np.maximum(0.0, pc_new_sum * inv2n - es * size_c_new)
                delta_U += weight[j] * ((ta_new - ta_old) + (tc_new - tc_old))

            delta = delta_Q - lambda_v * delta_U
            delta[a_pos] = 0.0
            best = int(np.argmax(delta))
            if float(delta[best]) > 1e-12:
                b = int(cand[best])
                clusters[i] = b
                sigma_tot[a] -= ki
                sigma_tot[b] += ki
                size[a] -= 1
                size[b] += 1
                for j in range(len(attrs)):
                    g = int(codes[j][i])
                    cnt[j][a, g] -= 1
                    cnt[j][b, g] += 1
                    phi[j][a] = cnt[j][a].astype(np.float64) - float(size[a]) * p_glob[j]
                    phi[j][b] = cnt[j][b].astype(np.float64) - float(size[b]) * p_glob[j]
                moves += 1
        if moves == 0:
            break
    _, remap = np.unique(clusters, return_inverse=True)
    return remap.astype(np.int64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p, "ce_fair_louvain")
    p.add_argument("--lambda", dest="lambda_v", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=0.10, help="TV tolerance of every penalized attribute")
    p.add_argument("--max-passes", type=int, default=5)
    p.add_argument("--attributes", nargs="+", default=None,
                   help="penalized columns of sensitive_attrs.csv (default: all audited attributes)")
    a = common.resolve_common_args(p.parse_args(), "ce_fair_louvain")
    cols = a.attributes or common.SENSITIVE_COLUMNS[a.dataset]

    t0 = time.time()
    H = common.load_incidence(a.data_dir)
    n = H.shape[1]
    sens = common.load_table(a.data_dir, "sensitive_attrs.csv", n)
    attrs = []
    for c in cols:
        codes = common.factorize(sens[c])
        attrs.append({"codes": codes, "num_groups": int(codes.max()) + 1,
                      "epsilon": a.epsilon, "weight": 1.0})
    G = build_clique_expansion(common.units_from_incidence(H), n)
    G.setdiag(0.0)
    G.eliminate_zeros()
    G = G.astype(np.float64).tocsr()
    clusters = fair_louvain_local_moves(G, np.arange(n, dtype=np.int64), attrs,
                                        lambda_v=a.lambda_v, max_passes=a.max_passes, seed=a.seed)
    common.write_partition(a.out, clusters)
    common.report("ce_fair_louvain", a.dataset, clusters, a.k, a.out)
    print(f"[ce_fair_louvain] {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
