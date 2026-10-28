# Data

The paper uses six attributed hypergraphs (Table 3) plus one large IMDb
hypergraph for the DuckDB scale-up. Three of them can be redistributed and
ship in this folder. The other four must be downloaded from their providers
and built locally, because their licences do not allow redistribution.

| Folder | Paper name | Ships here | Source | Licence |
|---|---|---|---|---|
| `openalex_db/` | OA-DB | yes | OpenAlex works API | CC0 |
| `openalex_cs_size3plus/` | OA-CS-3+ | yes | OpenAlex works API | CC0 |
| `acs_pums_household/` | ACS-PUMS-HH | yes | U.S. Census Bureau, ACS PUMS 2023 1-year, Massachusetts | Public domain |
| `movielens_1m/` | ML-1M | no, build it | GroupLens MovieLens-1M | GroupLens licence, no redistribution |
| `pokec_hobby/` | Pokec | no, build it | SNAP soc-Pokec profiles | Research use with citation |
| `imdb_cast/` | IMDb-Cast | no, build it | IMDb non-commercial datasets | Non-commercial use only, no redistribution |
| `imdb_cast_full/` | (DuckDB scale-up only) | no, build it | same IMDb files | same |

`configs/datasets.json` lists, for each dataset, the audited attributes, the
label column, the allowed unit sizes, the expected numbers of nodes, units and
memberships, the source URLs, the licence, and the sha256 of every processed
file used in the paper.

## Files in each dataset folder

| File | Content |
|---|---|
| `incidence.npz` | `scipy.sparse` CSR matrix, units x nodes, entries 1 |
| `node_features.npz` | CSR matrix, nodes x features (not in `imdb_cast_full`) |
| `sensitive_attrs.csv` | `node_id` and the audited attributes |
| `labels.csv` | `node_id`, `label` (class label) |
| `node_mapping.csv` | `node_id` to the source identifier |
| `hyperedge_mapping.csv` | unit id to the source identifier |
| `metadata.json` | construction rules, filters, sha256 of the files above |
| `diagnostics.json`, `README.md` | summaries |

Load a dataset with `scipy.sparse.load_npz("data/<folder>/incidence.npz")`
and `pandas.read_csv("data/<folder>/sensitive_attrs.csv")`.

## Commands

Check the shipped datasets:

```bash
python data/check_data.py
```

Build the others (raw files go to `data/raw/`, which is not part of the
repository; each download prints the licence of its source first):

```bash
python data/download.py --dataset movielens_1m
python data/build/build_movielens_1m.py

python data/download.py --dataset pokec_hobby
python data/build/build_pokec_hobby.py

python data/download.py --dataset imdb_cast
python data/build/build_imdb_cast.py
python data/build/build_imdb_cast_full.py      # only for the DuckDB scale-up

python data/check_data.py
```

`python data/download.py --dataset all` fetches the MovieLens, Pokec, IMDb
and ACS PUMS files in one go. Every builder takes `--raw-dir` and
`--out-dir`; the defaults are `data/raw/<source>/` and `data/<folder>/`.
`check_data.py --data-root DIR` checks folders under another directory.
The builders need only numpy, scipy and pandas.

The shipped ACS-PUMS-HH folder can be rebuilt with
`python data/download.py --dataset acs_pums_household` and
`python data/build/build_acs_pums_household.py`. The OpenAlex builders
(`build_openalex_db.py`, `build_openalex_cs_size3plus.py`) document the
construction and run on a works cache from
`python data/download.py --dataset openalex_db` (or `openalex_cs_size3plus`).
That download needs a free OpenAlex API key in the environment variable
`OPENALEX_API_KEY`, and `--mailto` is optional. OpenAlex changes over time,
so a new query does not return the May 2026 snapshot used in the paper; use
the shipped folders to reproduce the paper.

`openalex_bio/` (OpenAlex works with the Biology concept C95444343, 3,764 nodes, 1,799 units, 9,279
memberships, May 2026 snapshot, CC0) also ships. It is not one of the six datasets of Table 3; the learned
reconstruction of `baselines/learned_reconstruction/` uses it as "another OpenAlex hypergraph" for training.
It was built like OA-DB with a concept filter in place of the venue filter; `check_data.py` does not check it.

The other two folders under `data/` are not datasets: `partitions/` holds the 30 audited partitions and the
AHRC partition of `imdb_cast_full`, and `witnesses/` holds the search outputs that `run_rq1.py` checks.

## Time and disk space

Measured on one desktop CPU core.

| Dataset | Download | Build time | Peak memory | Output |
|---|---|---|---|---|
| ML-1M | 6 MB zip (24 MB extracted) | 15 s | < 1 GB | 2.3 MB |
| Pokec | 436 MB `.gz` (read without unpacking) | about 1-2 min (estimate) | < 2 GB (estimate) | 15 MB |
| IMDb-Cast | 1.3 GB (three `.tsv.gz` files) | 2 min | 0.7 GB | 14 MB |
| imdb_cast_full | same files | 3 min | 1.4 GB | 120 MB |
| ACS-PUMS-HH | 19 MB zips (73 MB extracted) | 5 s | < 1 GB | 5 MB |

## Checksums and exact reproduction

`check_data.py` compares each processed file with the sha256 of the file used
in the paper and checks the node, unit and membership counts of Table 3. It
exits with status 1 on any failure.

* ML-1M and ACS-PUMS-HH are fixed releases; `download.py` confirms that the
  downloaded archives have the sha256 of the ones used in the paper, and the
  builders reproduce the paper files.
* IMDb refreshes its files daily. The paper used the snapshot of 2026-05-07,
  which the builders reproduce exactly; a later download gives different
  counts and checksums, and `check_data.py` reports the difference.
* Pokec (SNAP) is a fixed 2012 release.
* The npz files of IMDb-Cast, imdb_cast_full and ACS-PUMS-HH were written by a
  Python build that compresses with zlib-ng (CPython 3.14 on Windows). Other
  Python builds store the same arrays with slightly different compressed
  bytes; `check_data.py` then compares the decompressed arrays and prints
  `ok (same arrays, different zip container)`.
* If git converts line endings on checkout (`core.autocrlf`), CSV files change
  bytes but not content; `check_data.py` prints `ok (line endings differ)`.
  `--strict` turns both fallbacks into failures.

## Citations

Please cite the providers when you use these data: F. M. Harper and
J. A. Konstan, "The MovieLens Datasets: History and Context", ACM TiiS 2015;
L. Takac and M. Zabovsky, "Data Analysis in Public Social Networks", 2012
(SNAP Pokec); IMDb non-commercial datasets (https://developer.imdb.com/non-commercial-datasets/);
J. Priem, H. Piwowar and R. Orr, "OpenAlex", 2022; U.S. Census Bureau,
American Community Survey 2023 1-year Public Use Microdata Sample.
