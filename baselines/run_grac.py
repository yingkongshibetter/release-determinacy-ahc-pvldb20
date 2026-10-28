"""GRAC partition (Fanseu Kamhoua et al., CIKM 2021).

Uses the authors' mediator Laplacian (third_party/grac_cikm/Laplacian_HGCN.py,
MIT license, vendored unchanged) and re-implements the filtering loop of their
GRAC.py on our inputs, as in the runs that produced the audited partitions:
    A = (1 - alpha) I + alpha * Laplacian(H, X, mediators)    built once, from X
    repeat up to `power` times:
        X <- A X;  U = top-k left singular vectors of X (svds);
        labels = k-means(U, k, n_init 10);  intra = mean intra-cluster distance of X
        stop when intra decreased by at most `tol` in 2 iterations
    partition = labels of the last iteration
Differences from GRAC.py (see baselines/README.md): GRAC.py rebuilds the
mediator Laplacian from the filtered features in every iteration, while this
loop builds it once from the input features; GRAC.py reseeds numpy at every
iteration, this loop seeds it once before the Laplacian is built.

Example:
    python baselines/run_grac.py --dataset openalex_db
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import common

GRAC_DIR = common.THIRD_PARTY / "grac_cikm"
POWER = {"pokec_hobby": 10}  # the audited Pokec partition used power 10, all others 20


def square_dist(prelabel: np.ndarray, feature) -> tuple[float, float]:
    """Mirror of square_dist in GRAC.py (mean intra- and inter-cluster distance)."""
    if sp.issparse(feature):
        feature = feature.toarray()
    feature = np.asarray(feature)
    k = len(np.unique(prelabel))
    label = np.zeros((prelabel.shape[0], k))
    label[range(prelabel.shape[0]), prelabel] = 1
    onehot = label.T
    count = onehot.sum(1).reshape(k, 1)
    count[count == 0] = 1
    mean = onehot @ feature / count
    a2 = (onehot @ (feature * feature) / count).sum(1)
    pdist2 = np.asarray(a2[:, None] + a2[None, :] - 2 * mean @ mean.T)
    intra = pdist2.trace() / k
    inter = (pdist2.sum() - pdist2.trace()) / max(1, k * (k - 1))
    return intra, inter


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p, "grac")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--tol", type=float, default=0.5)
    p.add_argument("--power", type=int, default=None, help="maximum number of filter iterations "
                   "(default 20, and 10 on pokec_hobby as in the audited run)")
    p.add_argument("--mediators", type=int, default=1)
    p.add_argument("--normalize", default=None, choices=[None, "l1", "l2"])
    a = common.resolve_common_args(p.parse_args(), "grac")
    if a.power is None:
        a.power = POWER.get(a.dataset, 20)
    sys.path.insert(0, str(GRAC_DIR))
    from Laplacian_HGCN import Laplacian

    t0 = time.time()
    H = common.load_incidence(a.data_dir)
    X = common.load_features(a.data_dir)
    n = H.shape[1]
    if X.shape[0] != n:
        raise SystemExit(f"features have {X.shape[0]} rows, incidence has {n} nodes")
    k = int(a.k)
    E = {i: e for i, e in enumerate(common.units_from_incidence(H)) if len(e) >= 2}
    feats = X.astype(np.float32).toarray()

    np.random.seed(int(a.seed))
    adj = Laplacian(n, E, feats, a.mediators)
    adj = (1 - a.alpha) * sp.eye(adj.shape[0]) + a.alpha * adj
    feats_arr = np.asarray(feats, dtype=np.float32)

    intra_prev, count, max_count, last_pred = None, 0, 2, None
    for it in range(1, int(a.power) + 1):
        feats_arr = adj.dot(feats_arr)
        if a.normalize == "l2":
            trick = normalize(feats_arr, norm="l2", axis=1)
        elif a.normalize == "l1":
            trick = normalize(feats_arr, norm="l1", axis=1)
        else:
            trick = feats_arr
        try:
            u, _, _ = svds(sp.csr_matrix(trick).astype(np.float32),
                           k=min(k, max(2, min(trick.shape) - 2)), which="LM")
        except Exception as e:  # kept from the original loop
            print(f"[grac] SVD failure at iteration {it}: {e}", flush=True)
            break
        pred = KMeans(n_clusters=k, init="k-means++", random_state=int(a.seed), n_init=10).fit_predict(u)
        intra, _ = square_dist(pred, feats_arr)
        print(f"[grac] iteration {it}: intra {intra:.4f}", flush=True)
        last_pred = pred
        if intra_prev is not None and intra_prev - intra <= a.tol:
            count += 1
            if count >= max_count:
                break
        intra_prev = intra
    if last_pred is None:
        raise SystemExit("GRAC produced no clustering")
    clusters = np.asarray(last_pred, dtype=np.int64)
    common.write_partition(a.out, clusters)
    common.report("grac", a.dataset, clusters, a.k, a.out)
    print(f"[grac] {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
