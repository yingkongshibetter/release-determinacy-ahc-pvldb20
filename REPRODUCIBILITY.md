# Release Determinacy for Fairness Certification in Attributed Hypergraph Clustering

This repository holds the code and data of the PVLDB paper *Release Determinacy for Fairness Certification
in Attributed Hypergraph Clustering* (Xuemin Wang, Yuanjin Hao, Liang Chang). It audits partitions of
attributed hypergraphs with hyperedge group preservation (HGP), the membership-weighted within-unit
segregation of groups across clusters. It decides which releases (pairwise graphs, the random-walk release,
incidence) determine HGP, computes the conservative certificates U_P, U_Q and U_Q,50 from a held release,
and evaluates HGP exactly with one SQL query when a certificate leaves a budget open. From the 30 stored
partitions and the data, the scripts regenerate every release, budget, audit, certificate and result of
RQ1, RQ2 and Practicality. Every decision is made in exact rational arithmetic.

## Requirements

- Python 3.12 (tested with CPython 3.12.10 on Windows 11 and an Intel Core i9-14900KF with 32 GB RAM).
- `python -m pip install -r requirements.txt` (numpy 2.5.2, scipy 1.18.0, pandas 3.0.5, networkx 3.6.1,
  duckdb 2.0.0, psutil 7.2.2).
- Optional: `requirements-search.txt` (numba 0.64.0, CPython 3.14.3) for the same-group exchange search.
- Optional: `baselines/requirements.txt` to rerun the five clustering methods, and
  `baselines/learned_reconstruction/requirements-shyre.txt` for the learned reconstruction.

## Data

The paper uses six attributed hypergraphs (Table 3), plus `imdb_cast_full` for the DuckDB scale-up.
OA-DB, OA-CS-3+ and ACS-PUMS-HH ship in `data/` (OpenAlex is CC0; ACS PUMS is public domain), and so does
OpenAlex-Bio (`data/openalex_bio/`), on which the learned reconstruction is also trained.
ML-1M, Pokec and IMDb-Cast must be built locally, because their licences do not allow redistribution:

```
python data/download.py --dataset all       # MovieLens-1M, SNAP Pokec, IMDb, ACS PUMS -> data/raw/
python data/build/build_movielens_1m.py
python data/build/build_pokec_hobby.py
python data/build/build_imdb_cast.py
python data/build/build_imdb_cast_full.py   # only for the scale-up
python data/check_data.py                   # sha256 and Table 3 counts
```

Downloads total about 1.8 GB and building takes under five minutes. IMDb updates its files daily; the paper
used the 2026-05-07 snapshot, so a new download builds a slightly different IMDb-Cast, and `check_data.py`
reports the difference. See `data/README.md` for sources, licences and details.

The repository also stores
- `data/partitions/<dataset>/<method>.csv.gz`: the 30 audited partitions (`node_id,cluster`), one per method
  and dataset; `baselines/` documents how each was produced (`baselines/README.md`). On IMDb-Cast, CAHC
  exceeds memory in its first pretraining epoch, and its file is the k-means partition of the randomly
  initialized encoder (CAHC-init). `data/partitions/imdb_cast_full/ahrc.npz` is the AHRC partition of the
  scale-up hypergraph.
- `data/witnesses/`: the outputs of the two searches of RQ1 that took hours, so that their counts can be
  checked in minutes. `declared/` holds the 260 compatible inputs of Figure 3 (L1 and L2, low and high, for
  the 65 cells with overlapping units), stored as the units that differ from the data. `exchange/` holds the
  75 sequences of same-group exchanges. `run_rq1.py` rebuilds and checks every witness in exact arithmetic;
  `run_search.py` reruns the searches.

## Run an audit

```
python run_audit.py --dataset OA-DB --method AHRC --attribute country
python run_audit.py --dataset Pokec --method GRAC --attribute AGE --tau 0.6 --duckdb
```

The script prints true HGP, the verdict of each pairwise auditor (with misses and false alarms marked), each
certificate (certified, or undetermined and sent to exact evaluation), the scale of Dbar with `--tau`, and
the exact HGP from the SQL query on SQLite (and DuckDB). Without `--budget`, it replays the shared budget B.
The core package is `ahc_audit/` (see its `__init__.py` for one line per module).

## Reproduce the paper

```
python -m unittest discover -s tests -t .                 # 13 fast correctness tests, about 15 s
python run_cells.py --screened --partition-calibrated     # 85 cells and all budgets, about 30 min
python run_rq1.py                                         # Tables 3, 5, 6, Figures 3, 4, RQ1, about 7 min
python run_rq2.py                                         # RQ2, Table 7, request counts, about 2 min
python run_constructions.py                               # Sections 2 and 3 in exact arithmetic, 2 s
python run_scaling.py batch                               # Practicality: batch audits, about 4 min
python run_scaling.py incremental --datasets ML-1M Pokec IMDb-Cast   # about 2 min
python run_scaling.py scale-up --replicas 1 2 4 8 18 36   # heavy, see below
```

`run_cells.py` writes `outputs/cells.csv` with every statistic of every cell (exact rationals as strings) and
its budgets, and `outputs/budgets.csv`. The other scripts read it and write the files below into `outputs/`
(git-ignored), together with `rq1.txt` and `rq2.txt`, which list every number with its paper item.

| Paper item | Command | Output |
|---|---|---|
| Table 3 (nodes, units, memberships, K, 30% screen) | `run_rq1.py` | `table3.csv` |
| Table 4 (auditors and budgets) | `run_cells.py` | `budgets.csv`, `budgets_partition.csv`, `cells.csv` |
| Table 5 | `run_rq1.py` | `table5.csv` |
| Table 6 (all five rows) | `run_rq1.py` | `table6.csv` |
| Figure 3 | `run_rq1.py` | `figure3.csv` (`undetermined_L1`), `rq1/declared_witnesses.csv` |
| Figure 4 | `run_rq1.py` | `figure4.csv` |
| RQ1 bullet 1 (same-group exchanges) | `run_rq1.py` (checks `data/witnesses/exchange`) | `rq1.txt` lines `RQ1-1`, `rq1/exchange.csv` |
| RQ1 bullet 2 (declared sizes, 41 and 24 of 65, at least 18) | `run_rq1.py` | `RQ1-2`, `rq1/declared_L0.csv`, `figure3.csv` |
| RQ1 bullet 3 (68 of 75, 12 of 65 policies, ACS components, misses, x0.6) | `run_rq1.py` | `RQ1-3`, `rq1/policy.csv`, `rq1/acs_components.csv` |
| RQ1 bullet 4 (dissimilarity scale) | `run_rq1.py` | `RQ1-4`, `figure4.csv` |
| RQ1 bullet 5 (fixed 0.1) | `run_rq1.py` | `RQ1-5` |
| RQ1 bullet 6 (reconstruction auditor) | `run_cells.py`, `run_rq1.py` | `RQ1-6`, `reconstruction/*.json` |
| RQ1 bullet 6 (learned reconstruction) | `baselines/learned_reconstruction/` | see `baselines/README.md` |
| RQ1 bullet 7 (split shares) | `run_rq1.py` | `RQ1-7` |
| RQ1, further seeds | `baselines/run_seeds.py`, then `run_cells.py --partitions` | see Additional notes |
| RQ2 bullet 1 (62 of 69, 57, 54, 51 of 56, 13 and 30) | `run_rq2.py` | `rq2.txt` lines `RQ2-1` |
| RQ2 bullet 2 (requests 7, 2, 7, 11; 6% to 19%; 53%) | `run_rq2.py` | `RQ2-2` |
| RQ2 bullet 3 (1.12 vs 1.38; LP 512 of 540, 420 for P; 59.2 ms vs 26 us) | `run_rq2.py` | `RQ2-3`, `rq2/lp_small_instances.csv` |
| RQ2 bullet 4 (rounded and pruned releases, 55 at top-50) | `run_rq2.py` | `RQ2-4`, `rq2/inexact.csv` |
| Practicality: 9 of 15 and 6 of 15 pairs need no request | `run_rq2.py` | `rq2.txt` line `Practicality` |
| Practicality: release build, request bytes, batch times, 10,800 audits | `run_scaling.py batch` | `scaling/batch_summary.txt` |
| Practicality: per-move update times | `run_scaling.py incremental` | `scaling/incremental/summary.csv` |
| Practicality: DuckDB 2.05e8 memberships, slopes, U_P scan of T8 | `run_scaling.py scale-up` | `scaling/scale_up/scale_up_table.csv`, `scale_up_slopes.csv` |
| Appendix B, Table 7 | `run_rq2.py` | `table7.csv` |
| Corollaries 1-4, Theorem 3, Lemma 1, 129 of 256 | `run_constructions.py` | `constructions.txt` |

### Expected runtimes and hardware

All times are for the workstation above, one process. `run_cells.py` takes about 30 min: the maximal-clique
reconstruction of IMDb-Cast (about 6 min, 2 GB) and the enumeration on ML-1M, which stops at the cap of
20 million cliques (about 4 min, 4.6 GB); the budget stream (about 6 min); the partition-calibrated budgets
(about 9 min); and the 85 cells (about 4 min). Peak memory stays below 8 GB. Without `--partition-calibrated`
the count of Figure 3 under partition-calibrated budgets is skipped. Counts and exact values do not depend on
the hardware; times do (for example the 59.2 ms and 26 us per set of RQ2 bullet 3).

Heavy items, not needed for the tables above:
- `run_scaling.py scale-up --replicas 1 2 4 8 18 36` builds DuckDB files of up to 2.05e8 memberships
  (about 30 min; 8 threads and a 4 GB memory limit; up to 30 GB of disk, as the database files total 15 GB and
  the exact query at x36 spills 13 GB). The exact query at x36 took 79.7 s at a peak of 4.2 GiB in the paper.
  `run_scaling.py smoke` checks the same code on OA-DB in about 15 s.
- `run_search.py declared` (Figure 3): 300 s per cell, rung and direction, 14.5 CPU hours for the 65 cells.
  `run_search.py exchange` (needs numba): 300 s per cell, about 6 CPU hours. Both stop on a time cap, so a
  rerun on other hardware finds other witnesses; `run_rq1.py --witness-dir outputs/witnesses` checks them.
  Where the exchange search stops because no exchange improves HGP (as on OA-DB), a rerun with
  `requirements-search.txt` reproduces the stored sequence exactly.
- Trained CAHC on Pokec and ACS-PUMS-HH ran on a server with 22 CPUs and 110 GiB (6,215 s at 87 GiB and
  2,331 s at 56 GiB). The other partitions take seconds to 20 minutes each on the workstation.
- The learned reconstruction (SHyRe) takes under about an hour for OA-DB, OA-CS-3+ and Pokec.

## Additional Notes

- The shared budget B replays one random stream, `numpy.random.default_rng(2026)`, over the datasets and
  attributes in the order of `configs/audit.json`. The strata are the class labels of `labels.csv`. The
  replay reproduces the budgets of the paper to within 2e-13, and the self-calibrated budgets exactly.
- The reconstruction auditor is not run on ML-1M, whose enumeration exceeds the cap. On IMDb-Cast the
  reconstruction has 3.2e8 memberships, above the 5e7 used for self-calibration, so only the shared and fixed
  budgets and tau apply there.
- Seeds: 57 cells were rerun with four further seeds (`baselines/run_seeds.py`, about 5 h on the workstation).
  The seed partitions are not stored here. To audit one seed, copy its files into the layout of
  `data/partitions/` (`<dataset>/<method>.csv.gz`; the AHRC file must stay the stored one, because the shared
  budget is computed from it), copy `outputs/budgets.csv` into a new output folder, and run
  `run_cells.py --partitions <folder> --out <new folder> --skip-reconstruction` and `run_rq1.py --out <new folder>`.
- `baselines/README.md` records, for each clustering method, the upstream commit and where the code that
  produced the partitions differs from the authors' code.
- Decisions on float64 releases (T in double precision, pruned rows) use outward-rounded bounds and certify
  only when the bound is at most the budget minus 1e-9; every other decision compares exact rationals.
- Times of the relational prototype in the paper use SQLite 3.50.4 (bundled with CPython 3.14.3) and DuckDB
  2.0.0. Runtimes depend on the hardware; counts and exact values do not.

## Citation

```
@misc{wang2026release,
  title  = {Release Determinacy for Fairness Certification in Attributed Hypergraph Clustering},
  author = {Wang, Xuemin and Hao, Yuanjin and Chang, Liang},
  note   = {Submitted to Proceedings of the VLDB Endowment, Vol. 20},
  year   = {2026}
}
```

Xuemin Wang, Yuanjin Hao and Liang Chang are with the Guilin University of Electronic Technology.

## License

The code is released under the MIT License (`LICENSE`). The shipped OpenAlex data are CC0 and the ACS PUMS
data are in the U.S. public domain. MovieLens, Pokec and IMDb data are not redistributed; their licences apply
to the files that `data/download.py` fetches.
