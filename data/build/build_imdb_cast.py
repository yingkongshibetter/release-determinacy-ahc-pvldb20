"""Build IMDb-Cast: actors grouped by the cast of one feature film.

Inputs (``data/raw/imdb/``, from https://datasets.imdbws.com/):
    title.basics.tsv.gz, title.principals.tsv.gz, name.basics.tsv.gz

Construction (paper, Table 3):
    units       non-adult feature films (titleType == "movie", isAdult == 0)
                with startYear in [1990, 2025]
    members     principal cast credits with category actor or actress
    unit size   3 <= |cast| <= 50, before and after the actor cap
    nodes       the 80,000 actors with the most such films, among actors
                with at least 2 films; ties in film count are broken by the
                order in which actors first appear in title.principals,
                then by nconst
Audited attributes: birth_cohort (birthYear bucket) and career_stage (age at
the first in-scope film). IMDb has no gender field; none is inferred.
Label: dominant film genre over the actor's in-scope films (ties broken by
the larger genre name).
Features: per-actor (genre x decade) film counts, L1-normalised.

IMDb refreshes these files daily. The paper used the snapshot downloaded on
2026-05-07; a later snapshot gives a different dataset (check_data.py will
report the mismatch).

Usage:
    python data/build/build_imdb_cast.py [--raw-dir DIR] [--out-dir DIR]
Runtime about 2 minutes and under 1 GB of RAM on a desktop CPU.
"""
from __future__ import annotations

import argparse
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import common

NULL = "\\N"
# The paper files were written on Windows: CRLF CSVs, Windows zip header.
EOL = common.CRLF
ZIP_HOST = common.ZIP_HOST_WINDOWS


def _open_tsv(p: Path):
    return gzip.open(p, "rt", encoding="utf-8")


def _bucket_birth(year: int | None) -> str:
    if year is None:
        return "missing"
    if year < 1950:
        return "pre1950"
    if year < 1970:
        return "1950s60s"
    if year < 1990:
        return "1970s80s"
    return "post1990"


def _bucket_career(start_age: int | None) -> str:
    if start_age is None:
        return "missing"
    if start_age < 25:
        return "lt25"
    if start_age < 40:
        return "25_39"
    if start_age < 55:
        return "40_54"
    return "55plus"


def build(raw: Path, out: Path, *, year_min: int, year_max: int, min_cast: int,
          max_cast: int, min_actor_degree: int, top_actors: int) -> None:
    hint = "run: python data/download.py --dataset imdb_cast"
    basics = common.require(raw / "title.basics.tsv.gz", hint)
    principals = common.require(raw / "title.principals.tsv.gz", hint)
    names = common.require(raw / "name.basics.tsv.gz", hint)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/6] title.basics: feature films in the year window", flush=True)
    movie_meta: dict[str, int] = {}
    movie_genres: dict[str, list[str]] = {}
    with _open_tsv(basics) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {n: i for i, n in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(idx):
                continue
            if parts[idx["titleType"]] != "movie" or parts[idx["isAdult"]] != "0":
                continue
            sy = parts[idx["startYear"]]
            if sy == NULL:
                continue
            try:
                sy = int(sy)
            except ValueError:
                continue
            if sy < year_min or sy > year_max:
                continue
            tconst = parts[idx["tconst"]]
            genres = parts[idx["genres"]]
            if genres == NULL or genres == "":
                genres = "Unknown"
            movie_meta[tconst] = sy
            movie_genres[tconst] = genres.split(",")
    print(f"      films in scope: {len(movie_meta):,}", flush=True)

    print("[2/6] title.principals: actor and actress credits", flush=True)
    cast: dict[str, list[str]] = defaultdict(list)
    with _open_tsv(principals) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {n: i for i, n in enumerate(header)}
        n_cols, i_cat, i_t, i_n = len(idx), idx["category"], idx["tconst"], idx["nconst"]
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < n_cols or parts[i_cat] not in ("actor", "actress"):
                continue
            tconst = parts[i_t]
            if tconst in movie_meta:
                cast[tconst].append(parts[i_n])

    print(f"[3/6] cast size in [{min_cast}, {max_cast}]", flush=True)
    cast = {t: c for t, c in cast.items() if min_cast <= len(c) <= max_cast}
    print(f"      films: {len(cast):,}", flush=True)

    print(f"[4/6] actors with >= {min_actor_degree} films, top {top_actors:,}", flush=True)
    actor_deg: dict[str, int] = defaultdict(int)
    first_seen: dict[str, int] = {}
    for film_index, c in enumerate(cast.values()):
        for nconst in set(c):
            actor_deg[nconst] += 1
            first_seen.setdefault(nconst, film_index)
    keep = {n for n, d in actor_deg.items() if d >= min_actor_degree}
    if top_actors and len(keep) > top_actors:
        ranked = sorted(actor_deg, key=lambda n: (-actor_deg[n], first_seen[n], n))
        keep = set(ranked[:top_actors])
    cast = {t: sorted({n for n in c if n in keep}) for t, c in cast.items()}
    cast = {t: c for t, c in cast.items() if min_cast <= len(c) <= max_cast}
    print(f"      actors: {len(keep):,}; films after the cap: {len(cast):,}", flush=True)

    print("[5/6] name.basics: birth years", flush=True)
    birth: dict[str, int | None] = {}
    with _open_tsv(names) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {n: i for i, n in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(idx):
                continue
            nconst = parts[idx["nconst"]]
            if nconst not in keep:
                continue
            try:
                birth[nconst] = int(parts[idx["birthYear"]])
            except ValueError:
                birth[nconst] = None

    print("[6/6] incidence, features, labels, attributes", flush=True)
    actors = sorted(keep)
    actor_to_id = {a: i for i, a in enumerate(actors)}
    films = sorted(cast, key=lambda t: (movie_meta[t], t))
    film_to_id = {t: i for i, t in enumerate(films)}
    n_actors, n_films = len(actors), len(films)

    rows, cols = [], []
    for tconst, c in cast.items():
        for nconst in c:
            rows.append(film_to_id[tconst])
            cols.append(actor_to_id[nconst])
    incidence = sp.csr_matrix((np.ones(len(rows), dtype=np.float32),
                               (np.asarray(rows), np.asarray(cols))),
                              shape=(n_films, n_actors))

    all_genres = sorted({g for gs in movie_genres.values() for g in gs})
    genre_to_idx = {g: i for i, g in enumerate(all_genres)}
    decades = sorted({(y // 10) * 10 for y in movie_meta.values()})
    decade_to_idx = {d: i for i, d in enumerate(decades)}
    n_feat = len(all_genres) * len(decades)

    genre_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    feat_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    first_year: dict[str, int] = {}
    for tconst, c in cast.items():
        sy = movie_meta[tconst]
        dec = decade_to_idx[(sy // 10) * 10]
        gs = movie_genres[tconst]
        for nconst in c:
            for g in gs:
                genre_counts[nconst][g] += 1
                feat_counts[nconst][genre_to_idx[g] * len(decades) + dec] += 1
            if nconst not in first_year or sy < first_year[nconst]:
                first_year[nconst] = sy
    f_rows, f_cols, f_vals = [], [], []
    for nconst, kd in feat_counts.items():
        for k, v in kd.items():
            f_rows.append(actor_to_id[nconst])
            f_cols.append(k)
            f_vals.append(v)
    features = sp.csr_matrix((np.asarray(f_vals, dtype=np.float32),
                              (np.asarray(f_rows), np.asarray(f_cols))),
                             shape=(n_actors, n_feat))
    row_sum = np.asarray(features.sum(axis=1)).flatten()
    with np.errstate(divide="ignore"):
        inv = np.where(row_sum > 0, 1.0 / row_sum, 0.0)
    features = sp.diags(inv) @ features

    label = []
    for nconst in actors:
        gc = genre_counts.get(nconst, {})
        label.append(max(gc.keys(), key=lambda g: (gc[g], g)) if gc else "Unknown")
    labels = pd.DataFrame({"node_id": np.arange(n_actors), "label": label})

    birth_cohort, career = [], []
    for nconst in actors:
        by = birth.get(nconst)
        birth_cohort.append(_bucket_birth(by))
        fy = first_year.get(nconst)
        career.append(_bucket_career(fy - by) if by is not None and fy is not None
                      else "missing")
    sens = pd.DataFrame({"node_id": np.arange(n_actors), "birth_cohort": birth_cohort,
                         "career_stage": career})

    common.save_npz(out / "incidence.npz", incidence, zip_host=ZIP_HOST)
    common.save_npz(out / "node_features.npz", features, zip_host=ZIP_HOST)
    common.write_csv(sens, out / "sensitive_attrs.csv", eol=EOL)
    common.write_csv(labels, out / "labels.csv", eol=EOL)
    common.write_csv(pd.DataFrame({"node_id": np.arange(n_actors), "imdb_nconst": actors}),
                     out / "node_mapping.csv", eol=EOL)
    common.write_csv(pd.DataFrame({"edge_id": np.arange(n_films), "imdb_tconst": films,
                                   "year": [movie_meta[t] for t in films],
                                   "genres": [",".join(movie_genres[t]) for t in films]}),
                     out / "hyperedge_mapping.csv", eol=EOL)

    sizes = np.diff(incidence.indptr)
    metadata = {
        "dataset_name": "imdb_cast",
        "paper_name": "IMDb-Cast",
        "raw_data_sources": [{"name": "IMDb non-commercial datasets",
                              "url": "https://datasets.imdbws.com/",
                              "snapshot_used_in_paper": "2026-05-07"}],
        "construction_date": common.utc_now(),
        "random_seed": common.RANDOM_SEED,
        "node_count": int(n_actors),
        "hyperedge_count": int(n_films),
        "volume": int(incidence.nnz),
        "feature_dim": int(n_feat),
        "sensitive_attributes": ["birth_cohort", "career_stage"],
        "label_source": "dominant genre over the actor's in-scope films",
        "hyperedge_rule": (
            f"one unit per film with titleType=='movie', isAdult==0, startYear in "
            f"[{year_min},{year_max}], cast size in [{min_cast},{max_cast}]; nodes are the "
            f"top {top_actors} actors by film count among actors with >= {min_actor_degree} films"),
        "filters": {"year_min": year_min, "year_max": year_max, "min_cast": min_cast,
                    "max_cast": max_cast, "min_actor_degree": min_actor_degree,
                    "top_actors": top_actors,
                    "tie_break": "film count desc, first appearance in title.principals, nconst"},
        "min_hyperedge_size": int(sizes.min()),
        "max_hyperedge_size": int(sizes.max()),
        "feature_genres": all_genres,
        "feature_decades": decades,
    }
    readme = (
        "# imdb_cast (IMDb-Cast)\n\n"
        "Units are the principal casts of non-adult feature films released 1990-2025 "
        f"(3 to {max_cast} credited actors); nodes are the {top_actors:,} actors with the most such films.\n"
        f"Nodes: {n_actors}. Units: {n_films}. Memberships: {incidence.nnz}.\n"
        "Audited attributes: birth_cohort, career_stage. Label: dominant film genre.\n"
        "Features: actor (genre x decade) film counts, L1-normalised.\n\n"
        "Built from the IMDb non-commercial datasets (https://datasets.imdbws.com/). "
        "Personal and non-commercial use only; do not redistribute.\n"
    )
    common.finish(out, metadata, readme, common.basic_diagnostics(incidence, sens, labels))
    print(f"imdb_cast: nodes={n_actors} units={n_films} memberships={incidence.nnz} -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", type=Path, default=common.RAW_DIR / "imdb")
    ap.add_argument("--out-dir", type=Path, default=common.DATA_DIR / "imdb_cast")
    ap.add_argument("--year-min", type=int, default=1990)
    ap.add_argument("--year-max", type=int, default=2025)
    ap.add_argument("--min-cast", type=int, default=3)
    ap.add_argument("--max-cast", type=int, default=50)
    ap.add_argument("--min-actor-degree", type=int, default=2)
    ap.add_argument("--top-actors", type=int, default=80000)
    a = ap.parse_args()
    build(a.raw_dir, a.out_dir, year_min=a.year_min, year_max=a.year_max,
          min_cast=a.min_cast, max_cast=a.max_cast, min_actor_degree=a.min_actor_degree,
          top_actors=a.top_actors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
