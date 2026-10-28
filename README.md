# Release Determinacy for Fairness Certification in Attributed Hypergraph Clustering

This repository provides the implementation of the paper **“Release Determinacy for Fairness Certification in Attributed Hypergraph Clustering”** by Xuemin Wang, Yuanjin Hao, and Liang Chang, submitted to PVLDB Vol. 20.

## Download

Click **Code → Download ZIP** above to download the complete source code and bundled data, or clone this repository:

```bash
git clone https://github.com/yingkongshibetter/release-determinacy-ahc-pvldb20.git
cd release-determinacy-ahc-pvldb20
```

Run the following commands from the repository directory. The [submission snapshot](https://github.com/yingkongshibetter/release-determinacy-ahc-pvldb20/releases/tag/pvldb20-submission) is also available as a release package.

## Requirements

Python 3.12. To install the dependencies:

```bash
pip install -r requirements.txt
```

## Prepare datasets

OpenAlex and ACS-PUMS-HH data are included in the package. To prepare MovieLens-1M, Pokec, and IMDb-Cast:

```bash
python data/download.py --dataset all
python data/build/build_movielens_1m.py
python data/build/build_pokec_hobby.py
python data/build/build_imdb_cast.py
python data/check_data.py
```

Dataset sources, licenses, and snapshot details are documented in `data/README.md` in this repository.

## Run a fairness audit

To audit a stored clustering partition:

```bash
python run_audit.py --dataset OA-DB --method AHRC --attribute country
```

## Reproduce experiments

To compute the audit cells and reproduce RQ1 and RQ2:

```bash
python run_cells.py --screened --partition-calibrated
python run_rq1.py
python run_rq2.py
```

To verify the theoretical constructions and run the batch audit experiment:

```bash
python run_constructions.py
python run_scaling.py batch
```

Results are saved to `outputs/`. See [the full reproducibility guide](REPRODUCIBILITY.md) for the paper-to-script mapping, tests, optional dependencies, additional experiments, and resource requirements.

## Additional Notes

- The package includes stored clustering partitions and search witnesses. Rerunning the clustering methods and searches is documented in `baselines/README.md` and the full reproducibility guide.
- IMDb data change over time. The paper used the 2026-05-07 snapshot; a fresh download may produce different counts.
- The code is released under the [MIT License](LICENSE). Dataset licenses are documented separately.
