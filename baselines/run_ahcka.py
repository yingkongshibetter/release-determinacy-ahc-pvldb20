"""AHCKA partition (Li, Yang, Shi, PACMMOD 2023), authors' code.

Calls cluster() of the authors' cluster.py (fetched into third_party/ahcka by
fetch_third_party.py) on the incidence matrix and the sparse node features,
with the authors' defaults: knn_k 10, alpha 0.2, beta 0.5, tmax 200,
init_iter 25, cluster interval 5, conductance early stop, exact cosine kNN.

The returned partition is the label vector of the LAST discretization that
cluster() computes before it stops, captured by wrapping spectral.discretize.
This is how the audited partitions were taken. cluster() itself keeps the
discretization with the lowest conductance and only uses it for its printed
scores; the two differ when the conductance early stop fires (on OA-DB and
ML-1M their ARI is about 0.6). --select min-conductance returns the latter.

AHCKA is deterministic here (random_init is off); --seed is recorded but has no effect.

Example:
    python baselines/run_ahcka.py --dataset openalex_db
"""
from __future__ import annotations

import argparse
import sys
import time
import types

import numpy as np

import common

AHCKA_DIR = common.THIRD_PARTY / "ahcka"


def import_ahcka():
    if not (AHCKA_DIR / "cluster.py").exists():
        raise SystemExit(f"AHCKA code not found in {AHCKA_DIR}; run: python baselines/fetch_third_party.py ahcka")
    sys.path.insert(0, str(AHCKA_DIR))
    try:
        import resource  # noqa: F401  (Linux and macOS)
    except ImportError:  # Windows: cluster.py only uses it to print peak memory
        res = types.ModuleType("resource")
        res.RUSAGE_SELF = 0
        res.getrusage = lambda _who: types.SimpleNamespace(ru_maxrss=0)
        sys.modules["resource"] = res
    import config
    import cluster
    return config, cluster


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p, "ahcka")
    p.add_argument("--tmax", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--init-iter", type=int, default=25)
    p.add_argument("--select", choices=["last", "min-conductance"], default="last",
                   help="last: last discretization, as in the audited partitions (default); "
                        "min-conductance: the discretization cluster() itself keeps")
    a = common.resolve_common_args(p.parse_args(), "ahcka")
    config, cluster_mod = import_ahcka()

    t0 = time.time()
    H = common.load_incidence(a.data_dir)          # units x nodes, the layout cluster() expects
    X = common.load_features(a.data_dir)
    n = H.shape[1]
    if X.shape[0] != n:
        raise SystemExit(f"features have {X.shape[0]} rows, incidence has {n} nodes")
    labels = common.factorize(common.load_table(a.data_dir, "labels.csv", n)["label"])
    knn_k = min(a.knn_k, n - 1)
    d_vec = np.asarray(H.sum(axis=0)).flatten()    # node degrees
    deg_dict = {i: float(d_vec[i]) for i in range(n)}

    config.labels = labels            # only used by cluster() for its printed scores
    config.data = "canonical"
    config.dataset = a.dataset
    config.seeds = int(a.seed)
    config.alpha = float(a.alpha)
    config.beta = float(a.beta)
    config.tmax = int(a.tmax)
    config.q_epsilon = 0.005
    config.mag_const = 1.0
    config.knn_k = int(knn_k)
    config.num_hop = 3
    config.init_iter = int(a.init_iter)
    config.approx_knn = False
    config.cluster_interval = 5
    config.cond_early_stop = True
    config.plot_performance = False
    config.verbose = False
    config.random_init = False
    config.hg_adj = H
    config.features = X.copy()

    last = {}
    original_discretize = cluster_mod.discretize

    def capture(eig):
        out = original_discretize(eig)
        last["pred"] = out[0]
        return out

    original_metrics = cluster_mod.clustering_metrics

    class CaptureBest(original_metrics):  # cluster() evaluates the partition it keeps
        def __init__(self, true_label, predict_label):
            last["best"] = predict_label
            super().__init__(true_label, predict_label)

    cluster_mod.discretize = capture
    cluster_mod.clustering_metrics = CaptureBest
    try:
        cluster_mod.cluster(H, X, a.k, deg_dict, alpha=a.alpha, beta=a.beta, tmax=a.tmax, ri=False)
    except KeyError as e:
        # cluster() ends with the authors' accuracy evaluation, which raises
        # KeyError when the partition has fewer clusters than label classes.
        # The partition has been captured before that point.
        print(f"[ahcka] evaluation in cluster() raised KeyError {e}; partition already captured", flush=True)
    finally:
        cluster_mod.discretize = original_discretize
        cluster_mod.clustering_metrics = original_metrics
    key = "pred" if a.select == "last" else "best"
    if key not in last:
        raise SystemExit("AHCKA cluster() returned before the partition was captured")
    clusters = np.asarray(last[key], dtype=np.int64).flatten()
    if clusters.size != n:
        raise SystemExit(f"AHCKA returned {clusters.size} labels for {n} nodes")
    common.write_partition(a.out, clusters)
    common.report("ahcka", a.dataset, clusters, a.k, a.out)
    print(f"[ahcka] {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
