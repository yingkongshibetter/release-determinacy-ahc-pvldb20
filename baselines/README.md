# Baselines: how the 30 audited partitions were produced

This folder holds the scripts that produced the 30 partitions the paper audits
(five methods on six datasets), the reruns with four further seeds, and the
learned hypergraph reconstruction of Wang and Kleinberg (ICLR 2024).
Each script reads one dataset folder and writes one partition file.

```
baselines/
  README.md
  requirements.txt             packages of the environment that produced the partitions
  common.py                    paths, per-dataset settings (K, seeds, attributes), file helpers
  fetch_third_party.py         downloads upstream code at the recorded commits
  run_ahrc.py                  AHRC (local re-implementation)
  run_ahcka.py                 AHCKA (authors' code, fetched)
  run_grac.py                  GRAC (authors' Laplacian, vendored; loop re-implemented)
  run_cahc.py                  CAHC (authors' model code, fetched)
  run_ce_fair_louvain.py       CE+Fair-Louvain (local implementation)
  run_seeds.py                 four further seeds for the 57 seed cells
  learned_reconstruction/
    requirements-shyre.txt     separate environment for SHyRe
    prepare_runs.py            training and query files
    run_shyre.py               one SHyRe run (authors' code, fetched)
    run_experiment.py          beta tuning and all runs of one dataset
    audit_reconstruction.py    HGP and verdicts on the reconstructed units
  third_party/
    .gitignore                 fetched code is not committed
    grac_cikm/                 Laplacian_HGCN.py and LICENSE (MIT), unchanged
```

## 1. Setup

The partitions were computed with CPython 3.14.3 on Windows, on the CPU.

```
python -m pip install -r baselines/requirements.txt
python baselines/fetch_third_party.py
```

`fetch_third_party.py` downloads AHCKA, CAHC and SHyRe as GitHub archives of
the commits below into `baselines/third_party/`, and checks the SHA-256 of
every file the scripts import. It uses only the Python standard library.

Each dataset folder `data/<dataset>/` holds `incidence.npz` (scipy sparse,
units x nodes), `node_features.npz`, `sensitive_attrs.csv` and `labels.csv`.
The dataset names are `movielens_1m` (ML-1M), `openalex_db` (OA-DB),
`openalex_cs_size3plus` (OA-CS-3+), `pokec_hobby` (Pokec), `imdb_cast`
(IMDb-Cast) and `acs_pums_household` (ACS-PUMS-HH).

## 2. Producing the 30 partitions

Every script takes the same options.

| Option | Default |
|---|---|
| `--dataset` | required |
| `--data-dir` | `data/<dataset>` |
| `--seed` | the seed of the audited partition |
| `--k` | K of Table 3 |
| `--out` | `outputs/partitions/<method>/<dataset>.csv`, columns `node_id,cluster` |

The defaults reproduce the settings of the audited runs, so one command per
partition is enough:

```
python baselines/run_ahrc.py            --dataset <dataset>
python baselines/run_ahcka.py           --dataset <dataset>
python baselines/run_grac.py            --dataset <dataset>
python baselines/run_cahc.py            --dataset <dataset>
python baselines/run_ce_fair_louvain.py --dataset <dataset>
```

The method folders are `ahrc`, `ahcka`, `grac`, `cahc` and `ce_fair_louvain`.
The table gives K and the seed of each partition. AHCKA, GRAC and CAHC take K
as input. AHRC and CE+Fair-Louvain return their natural number of clusters;
for them the script only reports whether the result has the K of Table 3.

| Dataset | AHRC | AHCKA | GRAC | CAHC | CE+FL |
|---|---|---|---|---|---|
| ML-1M | 11 (seed 42) | 18 (2026) | 18 (2026) | 18 (42) | 1,118 (42) |
| OA-DB | 8 (2026) | 11 (2026) | 11 (2026) | 11 (42) | 135 (42) |
| OA-CS-3+ | 34 (42) | 26 (2026) | 26 (2026) | 26 (42) | 2,478 (42) |
| Pokec | 34 (42) | 6 (2026) | 6 (2026), `--power 10` | 34 (42) | 5,733 (42) |
| IMDb-Cast | 485 (42) | 23 (2026) | 23 (2026) | 23 (42), `--mode init-only` | 6,658 (42) |
| ACS-PUMS-HH | 26 (2026) | 9 (2026) | 9 (2026) | 9 (42), 40 + 30 epochs | 22,706 (42) |

The flags in the table are already the defaults for that dataset. For
AHCKA, GRAC and CAHC, K is the number of class labels, except CAHC on Pokec,
where the original launcher set K = 34 by hand (the AHRC K on that dataset;
Pokec has 6 education classes).

## 3. Methods

The commit of each upstream copy was read from its download archive or git
clone, and the local copies equal those commits file for file.

| Method | Upstream code | Commit | License | In this repository |
|---|---|---|---|---|
| AHRC | https://github.com/ForPaperSubmissions/Attributed_Hypergraph_Representation_for_Clustering_Code (public AHRC code) | ea765e295fb951b549300ff31b5ef1d61e271aec (inspected, not used) | no license file | re-implemented in `run_ahrc.py` |
| AHCKA | https://github.com/CyanideCentral/AHCKA | e65f459701ae61a5794a36eab2bbf02b1b4fe2e4 | no license file | fetched |
| GRAC | https://github.com/BarakeelFanseu/GRAC_CIKM | 1e3b70f910bd6e7d4178abd55043a741a4f276da | MIT | `Laplacian_HGCN.py` vendored; loop re-implemented in `run_grac.py` |
| CAHC | https://github.com/nilics/CAHC | e6975ea33b0734255137e7592f83f7fcd325405d | no license file | fetched |
| CE+Fair-Louvain | none | - | - | local code in `run_ce_fair_louvain.py` |
| SHyRe | https://github.com/Abel0828/supervised_hypergraph_reconstruction | d49c77ed2ef5d41d5c1707211fcf17e2a2283830 | no license file | fetched |

### AHRC

`run_ahrc.py` is a local re-implementation of the AHRC pipeline, written
because the sparsifier of the public code is a Cython module built for Linux.
It computes the random-walk matrix T of the hypergraph, the walk
ST = alpha * sum over l = 0..gamma of (1 - alpha)^l T^l with alpha = 0.2 and
gamma = 2, the attribute graph SA = M + M^T of the cosine 20-nearest-neighbour
graph M of the features, the integrated matrix S = sqrt(row_norm(ST) row_norm(SA)),
and Louvain on max(S, S^T) with igraph `community_multilevel` (resolution 1,
igraph's random generator seeded before the call).
It differs from the public code in four ways.
- Sparsification: each T^l keeps its 200 largest entries per row, in place
  of the spanning-forest sparsifier (tau = 3) of the public code.
- The kNN graph has 20 neighbours, not 10.
- S keeps its 100 largest entries per row.
- Louvain is igraph's on the symmetrized S, not scikit-network's on S.

The original runs on ML-1M, OA-CS-3+ and ACS-PUMS-HH also computed a grid of
kNN sizes (10, 20, 50) and row caps (50, 100, 200); the audited partition is
the (20, 100) one. Each Louvain call reseeds its generator, so the script runs
only that setting (byte-identical on ML-1M).

### AHCKA

`run_ahcka.py` calls `cluster()` of the authors' `cluster.py` with their
defaults: 10 neighbours, alpha 0.2, beta 0.5, at most 200 orthogonal
iterations, 25 initialization iterations, discretization every 5 iterations,
the conductance early stop, and exact cosine kNN. On Windows the script stubs
the Linux-only `resource` module, which `cluster()` only uses to print memory.
AHCKA is deterministic here; the seed has no effect.

The audited partition is the label vector of the last discretization before
`cluster()` stops. `cluster()` itself keeps the discretization with the
lowest conductance. The two differ: their ARI is 0.62 on OA-DB and 0.59 on
ML-1M. `--select min-conductance` writes the one `cluster()` keeps.

### GRAC

`run_grac.py` uses the authors' `Laplacian()` unchanged and re-implements the
loop of their `GRAC.py`. It filters X with A = 0.5 I + 0.5 W, where W is the
normalized adjacency of the graph with mediators that `Laplacian()` returns,
takes the top-K left singular vectors, and runs k-means (10 starts). It stops
the second time the mean intra-cluster distance falls by at most 0.5, or after
20 iterations (10 on Pokec). The partition is that of the last iteration.
It differs from `GRAC.py` in two ways. `GRAC.py` rebuilds the mediator
Laplacian from the filtered features in every iteration; the script builds it
once from the input features. `GRAC.py` reseeds numpy in every iteration; the
script seeds it once.

### CAHC

`run_cahc.py` imports the authors' encoder, losses, augmentor, prototypes and
`train_step` and runs the schedule of their `train.py`: pretraining, k-means
(20 starts) on the embeddings to set the prototypes, fine-tuning, and the
argmax over the prototypes. The audited runs used smaller settings than the
authors' defaults (embedding size 512, 180 + 120 epochs): embedding size 256,
one layer, 60 pretraining and 40 fine-tuning epochs (40 and 30 on
ACS-PUMS-HH), learning rate and weight decay 5e-4, tau 0.5, all loss weights 1,
feature masking and incidence dropping 0.4. The original launcher also set a
1500 s wall budget; it never stopped a run, and the script has no budget.

The contrastive loss forms n x n matrices, so training needs about
4 x n^2 x 4 bytes. On IMDb-Cast, CAHC exceeds memory in its first pretraining
epoch, and the audited partition, CAHC-init, is k-means (10 starts) on the
embeddings of the randomly initialized encoder. `--mode init-only` produces
it and is the default on `imdb_cast` (on OA-DB it equals the fallback path of
the original runner byte for byte). Trained CAHC on Pokec and ACS-PUMS-HH ran
on a server with 22 CPUs and 110 GiB. All other partitions ran on the
workstation of Section 6.

### CE+Fair-Louvain

`run_ce_fair_louvain.py` builds the clique expansion with the size-only
weight 1 / C(k, 2) per pair of a unit of size k and maximizes
modularity minus lambda times the sum over attributes of
sum_c (|c| / n) max(0, TV_c - 0.10), where TV_c is the total-variation distance
between the attribute distribution of cluster c and that of the dataset.
Settings: lambda 1, weight 1 for every audited attribute of the dataset
(including race and disability on ACS-PUMS-HH), singleton start, at most 5
passes of local moves in a seeded random node order. There is no aggregation
phase, which is why K is large. This code was written for this study; no code
released with the Fair-Louvain paper was used.

## 4. Reproduction checks

| Method | Rerun with the stored seed |
|---|---|
| AHRC | identical on ML-1M, OA-DB, OA-CS-3+, ACS-PUMS-HH. Pokec and IMDb-Cast were not rerun on the workstation; on another CPU, Pokec matched only after forcing AVX2 kernels, and IMDb-Cast gave ARI 0.949 (484 clusters) |
| AHCKA | identical on all six |
| GRAC | identical on ML-1M, OA-DB, OA-CS-3+; IMDb-Cast ARI 0.62 and ACS-PUMS-HH ARI 0.9999 to the stored partition |
| CAHC | not bit-reproducible on the CPU: OA-DB ARI 0.999 (losses equal to the stored log to about four decimals), ML-1M ARI 0.61 |
| CE+Fair-Louvain | identical on all six |

The scripts in this folder were checked against the stored partitions on
OA-DB (all five methods; AHRC, AHCKA, GRAC and CE+Fair-Louvain byte-identical,
CAHC ARI 0.999, seeds 1 to 4 identical to the stored reruns) and on ML-1M
(AHRC, AHCKA, GRAC and CE+Fair-Louvain byte-identical).

## 5. Four further seeds (57 cells)

The paper reruns the clustering of 57 cells with seeds 1, 2, 3 and 4. These are
the 23 method-dataset pairs below; the audited partition stands for the
stored seed.

| Method | Datasets |
|---|---|
| AHRC | ML-1M, OA-DB, OA-CS-3+, ACS-PUMS-HH |
| AHCKA | all six (the seed has no effect) |
| GRAC | ML-1M, OA-DB, OA-CS-3+, IMDb-Cast, ACS-PUMS-HH |
| CAHC | ML-1M, OA-DB |
| CE+Fair-Louvain | all six |

```
python baselines/run_seeds.py --dry-run     # the 92 commands
python baselines/run_seeds.py --jobs 2
```

Output: `outputs/partitions_seeds/<method>/<dataset>/seed<k>.csv` (change it
with `--out-root`). Each run is
the command of the audited partition with only `--seed` and `--out` changed.
The pairs left out had a peak memory above about 3 GB, which the workstation
could not spare at the time (AHRC on Pokec and IMDb-Cast, GRAC on Pokec), took
over 20 minutes and 11 GB (CAHC on OA-CS-3+), or had an out-of-memory partition
at the time (CAHC on Pokec, IMDb-Cast and ACS-PUMS-HH). The original reruns used two
parallel jobs, with CAHC alone, and took 4.8 hours for 115 runs (the 92 above
plus a rerun of each stored seed).

## 6. Hardware and runtime

All partitions except trained CAHC on Pokec and ACS-PUMS-HH were computed on
an Intel Core i9-14900KF workstation with 32 GB RAM. Times of the audited runs
in seconds, with peak memory in parentheses when above 3 GB:

| Method | ML-1M | OA-DB | OA-CS-3+ | Pokec | IMDb-Cast | ACS-PUMS-HH |
|---|---|---|---|---|---|---|
| AHRC | 9 | 2 | 16 | 116 (8 GB) | 187 (13 GB) | 88 |
| AHCKA | 2 | 0.2 | 14 | 37 | 73 | 40 |
| GRAC | 4 | 2 | 6 | 69 (16 GB) | 16 | 8 |
| CAHC | 1,177 | 59 | 1,297 (11 GB) | 6,215 (87 GiB, server) | init-only, not timed | 2,331 (56 GiB, server) |
| CE+FL | 4 | 1 | 8 | 29 | 26 | 21 |

AHRC times cover the stages of the audited setting. CE+FL times cover the
local moves, not building the clique expansion. CAHC-init on IMDb-Cast is one
forward pass of the encoder and k-means.

## 7. Learned reconstruction (SHyRe)

The learned reconstruction is SHyRe-count with the default MLP classifier,
run with the authors' code on OA-DB, OA-CS-3+ and Pokec (45 cells). It was not
run on ML-1M (maximal-clique enumeration, SHyRe's first step, exceeds the
clique cap), IMDb-Cast (18.6 million maximal cliques; SHyRe's pure-Python
sampler statistics are infeasible on the workstation) or ACS-PUMS-HH
(disjoint households, which maximal cliques already recover).

SHyRe needs its own environment (Python 3.11.9, numpy older than 1.24):

```
python -m pip install -r baselines/learned_reconstruction/requirements-shyre.txt   # in a new venv
python baselines/fetch_third_party.py shyre
python baselines/learned_reconstruction/prepare_runs.py
python baselines/learned_reconstruction/run_experiment.py --dataset openalex_db --python <venv python>
python baselines/learned_reconstruction/run_experiment.py --dataset openalex_cs_size3plus --python <venv python>
python baselines/learned_reconstruction/run_experiment.py --dataset pokec_hobby --python <venv python>
python baselines/learned_reconstruction/audit_reconstruction.py --budgets outputs/budgets.csv
```

Training. The units of a dataset are split at random into halves a and b
(seeds 0, 1, 2). Four regimes are run:
- P1: each half is reconstructed from its projection by a model trained on
  the other half, and the two reconstructions are joined.
- P2: half a is known exactly and half b is reconstructed by the model trained on a.
- P3: the model trained on half a reconstructs the projection of all units.
- P4: a model trained on all units of `openalex_bio` (another OpenAlex
  hypergraph built the same way) reconstructs the projection of all units
  (OA-DB and OA-CS-3+ only; it needs `data/openalex_bio/incidence.npz`).

P1 to P3 train on half the units of the audited hypergraph, and P4 trains on
another OpenAlex hypergraph. The sampling
budget beta is tuned on a 90 / 10 split of half a (seed 0) over
300 to 1,000,000; the audited runs selected 300 on both OpenAlex datasets,
3,000 on Pokec and 1,000,000 on `openalex_bio`.

Audit. `audit_reconstruction.py` computes HGP of every audited partition in
`data/partitions/` (or of a rerun in `outputs/partitions/`, with `--partitions`) on the
reconstructed units, with the same integer
kernel as the auditor, and on the true units. A cell passes when HGP is at
most the threshold. Verdicts use the mean HGP over the three seeds, at the
shared budget B (`outputs/budgets.csv` of `run_cells.py`, or any CSV with columns
`dataset, attribute, B`) and at 0.1.
The audited runs gave the following. P1 and P2 match every true verdict on
the 45 cells at both thresholds. P3 raises one false alarm at 0.1 on OA-DB.
P4 raises two false alarms at 0.1 on OA-DB, as many as the maximal-clique
reconstruction.

Cost on the workstation: about 6 s per run on OA-DB and OA-CS-3+, 2 to 3
minutes per run on Pokec, under 1 GB each; P4 takes 1 to 2 minutes per run on
OA-DB and about 6 minutes and 9 GB per run on OA-CS-3+. Each tuning point is
capped at 30 minutes. On OA-DB, these scripts reproduce the stored HGP of all
180 cell, regime and seed combinations exactly.
