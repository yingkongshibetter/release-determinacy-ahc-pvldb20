"""Run the authors' SHyRe pipeline on one training/query pair and save the reconstruction.

Runs in the SHyRe environment (Python 3.11, numpy < 1.24, see
requirements-shyre.txt); the code is fetched into baselines/third_party/shyre.

    python -W ignore run_shyre.py --work <run folder> --dataset <run name> --beta <int> --seed <int> --out <file.pkl>

<run folder>/data/<run name>/{train,test}.txt must exist (prepare_runs.py).
The steps are those of the authors' main.py: set_up, load_graphs,
compute_cliques, DataLoader, then train() and the first part of evaluate(),
reproduced line by line up to the line that logs "Our Performance". The rest
of evaluate() only runs the paper's own baselines and is skipped; the cdlib
import those baselines need is replaced by a stub that is never called.
Variant SHyRe-count (--features count, the code default) with the default
MLP classifier. Nothing in the method (sampler, features, classifier, beta,
epochs, learning rate) is changed.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import threading
import time
import types
from copy import deepcopy
from pathlib import Path

import numpy as np

SHYRE = Path(__file__).resolve().parents[1] / "third_party" / "shyre"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--beta", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--features", default="count")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if not (SHYRE / "main.py").exists():
        raise SystemExit(f"SHyRe code not found in {SHYRE}; run: python baselines/fetch_third_party.py shyre")
    out = a.out.resolve()
    os.chdir(a.work)  # SHyRe reads ./data/<name>/ and writes its caches and logs there
    sys.path.insert(0, str(SHYRE))
    cd = types.ModuleType("cdlib")
    cda = types.ModuleType("cdlib.algorithms")

    def _unused(*_x, **_y):
        raise RuntimeError("baseline not used")

    cda.demon = _unused
    cda.kclique = _unused
    cd.algorithms = cda
    sys.modules["cdlib"] = cd
    sys.modules["cdlib.algorithms"] = cda

    peak = [0]
    stop = [False]
    try:
        import psutil
        proc = psutil.Process()

        def poll():
            while not stop[0]:
                peak[0] = max(peak[0], proc.memory_info().rss)
                time.sleep(0.2)
        th = threading.Thread(target=poll, daemon=True)
        th.start()
    except ImportError:
        th = None

    sys.argv = ["main.py", "--dataset", a.dataset, "--beta", str(a.beta), "--seed", str(a.seed),
                "--features", a.features]
    import utils
    import cliques as cq
    import train as tr
    from model import get_model

    t0 = time.perf_counter()
    args, logger = utils.set_up()
    graphs = utils.load_graphs(args, logger)
    clq = cq.compute_cliques(graphs, args, logger)
    dataloader = tr.DataLoader(graphs, clq, args, logger)
    # train()
    model1 = get_model(args, dataloader)
    model2 = deepcopy(model1)
    (X_train1, y_train1), (X_train2, y_train2) = dataloader.split_train()
    if X_train1.shape[0] > 0:
        model1.fit(X_train1, y_train1)
    if X_train2.shape[0] > 0:
        model2.fit(X_train2, y_train2)
    # evaluate(), first part
    X_test = dataloader.X_test
    num_max = dataloader.get_num_max_candidates("test")
    y1, y2 = np.array([]), np.array([])
    if X_test[:num_max].shape[0] > 0:
        y1 = model1.predict(X_test[:num_max])
    if X_test[num_max:].shape[0] > 0:
        y2 = model2.predict(X_test[num_max:])
    y_hat = np.hstack((y1, y2))
    rec = set(c for pr, c in zip(y_hat, dataloader.cliques["final_cliques_test"]) if pr > 0.5)
    P, R, F1, J = tr.get_performance_wrt_ground_truth(rec, graphs["simplicies_test"])
    logger.info("Our Performance: precision {:.4f}, recall {:.4f}, f1 {:.4f} jaccard {:.4f}".format(P, R, F1, J))
    t_all = time.perf_counter() - t0
    stop[0] = True
    if th is not None:
        th.join()
    smp = clq["sampler"]
    res = dict(dataset=a.dataset, beta=int(a.beta), seed=int(a.seed), features=a.features,
               reconstruction=sorted(rec), candidates=list(dataloader.cliques["final_cliques_test"]),
               n_candidates=len(dataloader.cliques["final_cliques_test"]), n_max_candidates=num_max,
               truth_in_candidates=int(dataloader.y_test.sum()), n_truth=len(graphs["simplicies_test"]),
               n_train_units=len(graphs["simplicies_train"]),
               sampler_train_recall=smp.n_collected / len(graphs["simplicies_train"]),
               max_cliques_train=len(clq["max_cliques_train"]), max_cliques_test=len(clq["max_cliques_test"]),
               precision=P, recall=R, f1=F1, jaccard=J, t_total_s=t_all, peak_rss_GB=peak[0] / 1024 ** 3)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(res, f)
    print("RESULT", {k: v for k, v in res.items() if k not in ("reconstruction", "candidates")}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
