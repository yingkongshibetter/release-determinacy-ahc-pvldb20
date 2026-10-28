"""CAHC partition (Ni, Zeng, Mu, Lin, WWW 2026), authors' model code.

Imports the authors' encoder, losses, augmentor, prototypes and train_step
(fetched into third_party/cahc by fetch_third_party.py) and runs the training
schedule of their train.py on our inputs, on the CPU, as in the audited runs:
    pretraining (gamma = 0) -> k-means (n_init 20) on the embeddings sets the
    prototypes -> fine-tuning (gamma = 1) -> partition = argmax over prototypes.
Settings of the audited runs, which differ from the authors' defaults
(emb_dim 512, 180 + 120 epochs): emb_dim 256, one layer, 60 pretraining and
40 fine-tuning epochs (40 and 30 on acs_pums_household); lr 5e-4, weight decay
5e-4, tau 0.5, alpha = beta = gamma = 1, feature mask and edge drop 0.4.

--mode init-only skips training: k-means (n_init 10) on the embeddings of the
randomly initialized encoder. This is the audited partition on imdb_cast
(CAHC-init), where training exceeds memory in the first pretraining epoch, and
it is the default for that dataset only.

Memory: the contrastive loss forms an n x n matrix, so one pretraining epoch
needs about 4 n^2 x 4 bytes (about 87 GiB on pokec_hobby, 56 GiB on
acs_pums_household). Training is not bit-reproducible on the CPU (see README).

Example:
    python baselines/run_cahc.py --dataset openalex_db
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import common  # numpy, pandas, scipy first, then torch, then the CAHC modules, as in the audited runs
import torch
import torch.nn as nn
from torch import optim

CAHC_DIR = common.THIRD_PARTY / "cahc"
EPOCHS = {"acs_pums_household": (40, 30)}   # all other datasets: (60, 40)


def import_cahc():
    if not (CAHC_DIR / "model" / "train.py").exists():
        raise SystemExit(f"CAHC code not found in {CAHC_DIR}; run: python baselines/fetch_third_party.py cahc")
    sys.path.insert(0, str(CAHC_DIR))
    # Same import order as the audited runs. Importing model.train sets
    # OMP_NUM_THREADS=6 in os.environ, after torch has been loaded.
    from model.EnhancedHGNN import EnhancedHGNN
    from model.Loss import Loss
    from model.Augmentor import Augmentor
    from model.ClusteringLayer import ClusteringPrototypes
    from model.train import train_step, create_negative_samples, fix_seed
    return EnhancedHGNN, Loss, Augmentor, ClusteringPrototypes, train_step, create_negative_samples, fix_seed


def hyperedge_index(H) -> torch.Tensor:
    """CAHC's (2, memberships) index: row 0 node id, row 1 unit id."""
    rows, cols = [], []
    for e_idx, members in enumerate(common.units_from_incidence(H)):
        for v in members:
            rows.append(int(v))
            cols.append(int(e_idx))
    return torch.tensor([rows, cols], dtype=torch.long)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p, "cahc")
    p.add_argument("--mode", choices=["train", "init-only"], default=None,
                   help="default: init-only on imdb_cast, train elsewhere")
    p.add_argument("--pretrain-epochs", type=int, default=None)
    p.add_argument("--finetune-epochs", type=int, default=None)
    p.add_argument("--emb-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--feat-mask-rate", type=float, default=0.4)
    p.add_argument("--edge-drop-rate", type=float, default=0.4)
    a = common.resolve_common_args(p.parse_args(), "cahc")
    if a.mode is None:
        a.mode = "init-only" if a.dataset == "imdb_cast" else "train"
    pt, ft = EPOCHS.get(a.dataset, (60, 40))
    a.pretrain_epochs = pt if a.pretrain_epochs is None else a.pretrain_epochs
    a.finetune_epochs = ft if a.finetune_epochs is None else a.finetune_epochs
    (EnhancedHGNN, Loss, Augmentor, ClusteringPrototypes,
     train_step, create_negative_samples, fix_seed) = import_cahc()
    from sklearn.cluster import KMeans

    fix_seed(int(a.seed))
    device = torch.device("cpu")
    t0 = time.time()
    H = common.load_incidence(a.data_dir)
    n = H.shape[1]
    he_index = hyperedge_index(H).to(device)
    X = common.load_features(a.data_dir)
    if X.shape[0] != n:
        raise SystemExit(f"features have {X.shape[0]} rows, incidence has {n} nodes")
    features = torch.tensor(X.toarray(), dtype=torch.float32, device=device)
    k = int(a.k)
    print(f"[cahc] {a.dataset}: n={n} units={H.shape[0]} k={k} mode={a.mode} "
          f"epochs={a.pretrain_epochs}+{a.finetune_epochs}", flush=True)

    encoder = EnhancedHGNN(node_dim=features.shape[1], emb_dim=a.emb_dim, num_layers=a.num_layers).to(device)

    if a.mode == "init-only":
        encoder.eval()
        with torch.no_grad():
            emb, _ = encoder(features, he_index)
        clusters = KMeans(n_clusters=k, n_init=10, random_state=int(a.seed)).fit_predict(emb.cpu().numpy())
    else:
        cluster_module = ClusteringPrototypes(k, a.emb_dim, a.tau).to(device)
        project_head = nn.Sequential(nn.Linear(a.emb_dim, a.emb_dim), nn.ReLU(),
                                     nn.Linear(a.emb_dim, a.emb_dim)).to(device)
        loss_fn = Loss(a.emb_dim, a.emb_dim).to(device)
        augmentor = Augmentor(a.feat_mask_rate, a.edge_drop_rate)
        params = (list(encoder.parameters()) + list(project_head.parameters())
                  + list(loss_fn.parameters()) + list(cluster_module.parameters()))
        optimizer = optim.Adam(params, lr=a.lr, weight_decay=a.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10)

        def epoch(gamma: float) -> dict:
            neg = create_negative_samples(he_index, n)
            f1, h1, f2, h2 = augmentor.augment(features, he_index)
            return train_step(encoder, project_head, cluster_module, optimizer, loss_fn,
                              features, he_index, neg, f1, h1, f2, h2,
                              a.alpha, a.beta, gamma, a.tau)

        for e in range(a.pretrain_epochs):
            m = epoch(0.0)
            if (e + 1) % 10 == 0 or e == 0:
                print(f"[cahc] pretrain {e + 1}/{a.pretrain_epochs} loss={m['total_loss']:.4f}", flush=True)
        encoder.eval()
        with torch.no_grad():
            init_emb, _ = encoder(features, he_index)
        km = KMeans(n_clusters=k, n_init=20, random_state=int(a.seed)).fit(init_emb.cpu().numpy())
        cluster_module.prototypes.data = torch.tensor(km.cluster_centers_, dtype=torch.float, device=device)
        for e in range(a.finetune_epochs):
            m = epoch(a.gamma)
            scheduler.step(m["total_loss"])
            if (e + 1) % 10 == 0 or e == 0:
                print(f"[cahc] finetune {e + 1}/{a.finetune_epochs} loss={m['total_loss']:.4f}", flush=True)
        encoder.eval()
        cluster_module.eval()
        with torch.no_grad():
            node_emb, _ = encoder(features, he_index)
            clusters = torch.argmax(cluster_module(node_emb), dim=1).cpu().numpy()

    clusters = np.asarray(clusters, dtype=np.int64)
    common.write_partition(a.out, clusters)
    common.report("cahc", a.dataset, clusters, a.k, a.out)
    print(f"[cahc] {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
