"""Build imdb_cast_full: every film and TV series with three or more principal actors.

Used only for the DuckDB scale-up by disjoint copies (5.70M memberships in the
paper). Same IMDb files as IMDb-Cast (``data/raw/imdb/``).

Construction:
    units       titles with titleType in {movie, tvMovie, tvSeries} (any year,
                adult titles included) whose principal credits list at least 3
                actor/actress entries
    nodes       every actor or actress in such a unit
Attributes: gender (credit category: actress if credited more often as
actress than as actor, otherwise male), birth_cohort (birthYear bucket,
missing for most people). Label: dominant primary genre over the actor's
titles (first-seen genre on ties). No feature matrix is written.

IMDb refreshes these files daily; the paper used the 2026-05-07 snapshot.

Usage:
    python data/build/build_imdb_cast_full.py [--raw-dir DIR] [--out-dir DIR]
Runtime about 3 minutes and 1.5 GB of RAM on a desktop CPU.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import common

csv.field_size_limit(10 ** 7)
NULL = "\\N"
ZIP_HOST = common.ZIP_HOST_WINDOWS   # the paper file was written on Windows
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:.0f}s] {msg}", flush=True)


def cohort(y):
    if y is None:
        return "missing"
    if y < 1950:
        return "pre1950"
    if y < 1970:
        return "1950s60s"
    if y < 1990:
        return "1970s80s"
    return "post1990"


def build(raw: Path, out: Path) -> None:
    hint = "run: python data/download.py --dataset imdb_cast_full"
    basics = common.require(raw / "title.basics.tsv.gz", hint)
    principals = common.require(raw / "title.principals.tsv.gz", hint)
    names = common.require(raw / "name.basics.tsv.gz", hint)
    out.mkdir(parents=True, exist_ok=True)

    # The csv module (default quoting) is kept on purpose: it is what the
    # paper build used, and it decides how lines with stray quotes are read.
    title_genre: dict[str, str] = {}
    with gzip.open(basics, "rt", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if len(row) > 8 and row[1] in ("movie", "tvMovie", "tvSeries"):
                title_genre[row[0]] = row[8].split(",")[0] if row[8] not in ("", NULL) else "Unknown"
    log(f"titles: {len(title_genre):,}")

    cast: dict[str, list[str]] = defaultdict(list)
    actor_votes: dict[str, int] = defaultdict(int)
    actress_votes: dict[str, int] = defaultdict(int)
    with gzip.open(principals, "rt", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if len(row) > 3 and row[3] in ("actor", "actress") and row[0] in title_genre:
                cast[row[0]].append(row[2])
                if row[3] == "actor":
                    actor_votes[row[2]] += 1
                else:
                    actress_votes[row[2]] += 1
    log(f"titles with cast: {len(cast):,}")

    edges = {t: c for t, c in cast.items() if len(c) >= 3}
    del cast
    node_set: set[str] = set()
    for c in edges.values():
        node_set.update(c)
    log(f"units: {len(edges):,}, nodes: {len(node_set):,}")

    birth: dict[str, int] = {}
    with gzip.open(names, "rt", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if row[0] in node_set and len(row) > 2 and row[2] not in ("", NULL):
                try:
                    birth[row[0]] = int(row[2])
                except ValueError:
                    pass
    log(f"nodes with birthYear: {len(birth):,}")

    nodes = sorted(node_set)
    del node_set
    nidx = {nc: i for i, nc in enumerate(nodes)}
    titles = sorted(edges)
    n, m = len(nodes), len(titles)
    rows, cols = [], []
    for ei, t in enumerate(titles):
        for nc in set(edges[t]):
            rows.append(ei)
            cols.append(nidx[nc])
    H = sp.csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(m, n))
    del rows, cols
    common.save_npz(out / "incidence.npz", H, zip_host=ZIP_HOST)
    log(f"incidence: {H.shape}, nnz {H.nnz:,}")

    def gender(nc: str) -> str:
        return "female" if actress_votes.get(nc, 0) > actor_votes.get(nc, 0) else "male"

    # csv.writer ends rows with "\r\n" on every platform, as in the paper files.
    with open(out / "sensitive_attrs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "gender", "birth_cohort"])
        for nc in nodes:
            w.writerow([nidx[nc], gender(nc), cohort(birth.get(nc))])

    node_genre: dict[str, Counter] = defaultdict(Counter)
    for t in titles:
        g = title_genre[t]
        for nc in set(edges[t]):
            node_genre[nc][g] += 1
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "label"])
        for nc in nodes:
            w.writerow([nidx[nc], node_genre[nc].most_common(1)[0][0]
                        if node_genre[nc] else "Unknown"])
    with open(out / "node_mapping.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "nconst"])
        for nc in nodes:
            w.writerow([nidx[nc], nc])
    with open(out / "hyperedge_mapping.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["edge_id", "tconst"])
        for ei, t in enumerate(titles):
            w.writerow([ei, t])

    genders = Counter(gender(nc) for nc in nodes)
    sizes = np.diff(H.indptr)
    metadata = {
        "dataset_name": "imdb_cast_full",
        "raw_data_sources": [{"name": "IMDb non-commercial datasets",
                              "url": "https://datasets.imdbws.com/",
                              "snapshot_used_in_paper": "2026-05-07"}],
        "construction_date": common.utc_now(),
        "n_nodes": n, "n_edges": m, "incidence": int(H.nnz),
        "gender": dict(genders),
        "birth_coverage": round(len(birth) / n, 3),
        "edge_size_median": int(np.median(sizes)),
        "edge_size_max": int(sizes.max()),
        "hyperedge_rule": "titleType in {movie, tvMovie, tvSeries}; >= 3 actor/actress credits",
    }
    common.write_json(out / "metadata.json", dict(metadata, file_sha256=common.file_hashes(out)))
    (out / "README.md").write_text(
        "# imdb_cast_full\n\n"
        "Units are IMDb films, TV films and TV series with three or more principal actors; "
        "nodes are those actors.\n"
        f"Nodes: {n}. Units: {m}. Memberships: {H.nnz}.\n"
        "Used only for the DuckDB scale-up. Personal and non-commercial use only "
        "(IMDb non-commercial datasets); do not redistribute.\n", encoding="utf-8")
    log(f"done: nodes={n} units={m} memberships={H.nnz} -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", type=Path, default=common.RAW_DIR / "imdb")
    ap.add_argument("--out-dir", type=Path, default=common.DATA_DIR / "imdb_cast_full")
    a = ap.parse_args()
    build(a.raw_dir, a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
