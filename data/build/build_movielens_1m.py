"""Build ML-1M: MovieLens-1M users grouped by co-rating events.

Input: the MovieLens-1M release (users.dat, movies.dat, ratings.dat), fetched by
``python data/download.py --dataset movielens_1m`` into ``data/raw/movielens/``.

Construction (paper, Table 3):
    units       one (film, calendar month) cell: the users who rated that film
                with >= 4 stars in that month (UTC); 3 <= size <= 50, larger
                cells are dropped
    nodes       users that belong to at least one unit
Audited attributes: gender, age_bucket (users.dat).
Label: dominant film genre, i.e. the argmax of the user's rating-weighted genre
histogram; an integer index into ``feature_genres`` in metadata.json.
Features: that histogram over the 18 MovieLens genres, computed from all of the
user's ratings and L1-normalised. No demographic field enters the features.

Usage:
    python data/build/build_movielens_1m.py [--raw-dir DIR] [--out-dir DIR]
``--raw-dir`` may hold the three .dat files, an ``ml-1m/`` folder with them, or
``ml-1m.zip``. Runtime about 15 seconds.
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import common

AGE_BUCKETS = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49",
               50: "50-55", 56: "56+"}

# Line endings and zip host of the paper files: the dataset was built on
# Linux; labels.csv was rewritten later on Windows.
EOL_MAIN = common.LF
EOL_LABELS = common.CRLF
ZIP_HOST = common.ZIP_HOST_UNIX


def _source(raw_dir: Path, name: str):
    for p in (raw_dir / name, raw_dir / "ml-1m" / name):
        if p.exists():
            return p
    zpath = common.require(raw_dir / "ml-1m.zip",
                           "run: python data/download.py --dataset movielens_1m")
    with zipfile.ZipFile(zpath) as z:
        return io.StringIO(z.read("ml-1m/" + name).decode("latin-1"))


def _read(raw_dir: Path, name: str, names: list[str], dtype: dict) -> pd.DataFrame:
    return pd.read_csv(_source(raw_dir, name), sep="::", engine="python", header=None,
                       names=names, dtype=dtype, encoding="latin-1")


def _build_units(liked: pd.DataFrame, *, min_size: int, max_size: int):
    edges, meta = [], []
    stats = {"raw_cell_count": 0, "small_cell_count": 0, "oversized_cell_count": 0}
    for (movie_id, month), grp in liked.groupby(["movie_id", "month"], sort=True):
        stats["raw_cell_count"] += 1
        if len(grp) < min_size:
            stats["small_cell_count"] += 1
            continue
        members = grp.sort_values(["timestamp", "user_id"])["user_id"].tolist()
        if len(members) > max_size:
            stats["oversized_cell_count"] += 1
            continue
        edges.append(members)
        meta.append({"external_id": f"ml1m_m{int(movie_id)}_{month}",
                     "raw_size": len(members), "kept_size": len(members),
                     "construction_rule": "movie_month_liked_>=4"})
    return edges, meta, stats


def _genre_features(ratings, movies, n_users, user_to_node):
    genre_list = sorted({g.strip() for gs in movies["genre_list"] for g in gs if g.strip()})
    genre_idx = {g: i for i, g in enumerate(genre_list)}
    rows, cols, vals = [], [], []
    for _, row in movies.iterrows():
        for g in row["genre_list"]:
            g = g.strip()
            if g:
                rows.append(int(row["movie_id"]))
                cols.append(genre_idx[g])
                vals.append(1.0)
    movie_max = int(movies["movie_id"].max()) + 1
    movie_genre = sp.csr_matrix((np.asarray(vals, dtype=np.float32),
                                 (np.asarray(rows), np.asarray(cols))),
                                shape=(movie_max, len(genre_list)))
    users = ratings["user_id"].to_numpy()
    items = ratings["movie_id"].to_numpy()
    values = ratings["rating"].to_numpy(dtype=np.float32)
    keep = np.array([u in user_to_node for u in users.tolist()], dtype=bool)
    users, items, values = users[keep], items[keep], values[keep]
    nodes = np.array([user_to_node[u] for u in users.tolist()], dtype=np.int64)
    user_movie = sp.csr_matrix((values, (nodes, items)), shape=(n_users, movie_max))
    feat = (user_movie @ movie_genre).tocsr()
    row_sums = np.asarray(feat.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    feat = feat.multiply((1.0 / row_sums)[:, None]).tocsr().astype(np.float32)
    return feat, genre_list


def _dominant_genre(features: sp.csr_matrix) -> np.ndarray:
    """argmax of the row-normalised genre histogram (first index on ties)."""
    dense = features.toarray()
    row_sum = dense.sum(axis=1)
    normed = dense / np.where(row_sum > 0, row_sum, 1.0)[:, None]
    return np.argmax(normed, axis=1).astype(np.int64)


def build(raw_dir: Path, out_dir: Path, *, min_size: int, max_size: int,
          min_rating: int) -> None:
    users = _read(raw_dir, "users.dat", ["user_id", "gender", "age", "occupation", "zip"],
                  {"user_id": np.int64, "age": np.int64, "occupation": np.int64})
    users["age_bucket"] = users["age"].map(AGE_BUCKETS)
    movies = _read(raw_dir, "movies.dat", ["movie_id", "title", "genres"],
                   {"movie_id": np.int64})
    movies["genre_list"] = movies["genres"].fillna("").str.split("|")
    ratings = _read(raw_dir, "ratings.dat", ["user_id", "movie_id", "rating", "timestamp"],
                    {"user_id": np.int64, "movie_id": np.int64, "rating": np.int64,
                     "timestamp": np.int64})

    liked = ratings[ratings["rating"] >= min_rating].copy()
    liked["month"] = pd.to_datetime(liked["timestamp"], unit="s").dt.to_period("M")
    edges, edge_meta, stats = _build_units(liked, min_size=min_size, max_size=max_size)

    surviving = sorted({u for e in edges for u in e})
    user_to_node = {u: i for i, u in enumerate(surviving)}
    n_nodes, n_edges = len(surviving), len(edges)
    rows, cols = [], []
    for e_idx, members in enumerate(edges):
        for u in members:
            rows.append(e_idx)
            cols.append(user_to_node[u])
    incidence = sp.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)),
                              shape=(n_edges, n_nodes)).astype(np.int32, copy=False)

    features, genre_list = _genre_features(ratings, movies, n_nodes, user_to_node)
    survivors = users[users["user_id"].isin(list(user_to_node))].copy()
    survivors["node_id"] = survivors["user_id"].map(user_to_node)
    survivors = survivors.sort_values("node_id")

    sensitive = pd.DataFrame({"node_id": survivors["node_id"].to_numpy(),
                              "gender": survivors["gender"].to_numpy(),
                              "age_bucket": survivors["age_bucket"].to_numpy()})
    labels = pd.DataFrame({"node_id": np.arange(n_nodes), "label": _dominant_genre(features)})
    node_mapping = pd.DataFrame({"node_id": np.arange(n_nodes, dtype=np.int64),
                                 "external_id": survivors["user_id"].tolist()})
    hyperedge_mapping = pd.DataFrame({
        "hyperedge_id": np.arange(n_edges, dtype=np.int64),
        "external_id": [m["external_id"] for m in edge_meta],
        "source_type": "movie_month",
        "raw_size": [m["raw_size"] for m in edge_meta],
        "kept_size": [m["kept_size"] for m in edge_meta],
        "construction_rule": [m["construction_rule"] for m in edge_meta],
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    common.save_npz(out_dir / "incidence.npz", incidence, zip_host=ZIP_HOST)
    common.save_npz(out_dir / "node_features.npz", features, zip_host=ZIP_HOST)
    common.write_csv(sensitive.sort_values("node_id"), out_dir / "sensitive_attrs.csv",
                     eol=EOL_MAIN)
    common.write_csv(labels, out_dir / "labels.csv", eol=EOL_LABELS)
    common.write_csv(node_mapping, out_dir / "node_mapping.csv", eol=EOL_MAIN)
    common.write_csv(hyperedge_mapping, out_dir / "hyperedge_mapping.csv", eol=EOL_MAIN)

    metadata = {
        "dataset_name": "movielens_1m",
        "paper_name": "ML-1M",
        "raw_data_sources": [{"name": "MovieLens-1M",
                              "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip"}],
        "construction_date": common.utc_now(),
        "random_seed": common.RANDOM_SEED,
        "node_count": int(n_nodes),
        "hyperedge_count": int(n_edges),
        "volume": int(incidence.nnz),
        "feature_dim": int(features.shape[1]),
        "sensitive_attributes": ["gender", "age_bucket"],
        "label_source": "dominant film genre: argmax of the row-normalised genre "
                        "feature vector, as an index into feature_genres",
        "hyperedge_rule": f"users who rated the same movie >= {min_rating} in the same "
                          "calendar month (UTC); oversized cells dropped",
        "filters": {"min_rating_for_edge": min_rating, "min_size": min_size,
                    "max_size": max_size, "oversized_policy": "drop",
                    "feature_uses_all_ratings": True,
                    "feature_excludes_sensitive_attrs": True},
        "min_hyperedge_size": min_size,
        "max_hyperedge_size": max_size,
        "removed_node_count": int(len(users) - n_nodes),
        "removed_hyperedge_count": int(stats["small_cell_count"] + stats["oversized_cell_count"]),
        "raw_movie_month_cell_count": int(stats["raw_cell_count"]),
        "small_movie_month_cell_count": int(stats["small_cell_count"]),
        "oversized_movie_month_cell_count": int(stats["oversized_cell_count"]),
        "feature_genres": genre_list,
    }
    readme = (
        "# movielens_1m (ML-1M)\n\n"
        f"Units are users who rated the same film >= {min_rating} stars in the same calendar month, "
        f"size in [{min_size}, {max_size}].\n"
        f"Nodes: {n_nodes}. Units: {n_edges}. Memberships: {incidence.nnz}.\n"
        "Audited attributes: gender, age_bucket. Label: dominant film genre "
        "(index into feature_genres in metadata.json).\n"
        "Features: rating-weighted genre histogram, L1-normalised.\n\n"
        "Built from MovieLens-1M (GroupLens). The GroupLens licence does not allow "
        "redistribution without permission.\n"
    )
    common.finish(out_dir, metadata, readme,
                  common.basic_diagnostics(incidence, sensitive, labels))
    print(f"movielens_1m: nodes={n_nodes} units={n_edges} memberships={incidence.nnz} -> {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", type=Path, default=common.RAW_DIR / "movielens")
    ap.add_argument("--out-dir", type=Path, default=common.DATA_DIR / "movielens_1m")
    ap.add_argument("--min-edge-size", type=int, default=3)
    ap.add_argument("--max-edge-size", type=int, default=50)
    ap.add_argument("--min-rating-for-edge", type=int, default=4)
    a = ap.parse_args()
    build(a.raw_dir, a.out_dir, min_size=a.min_edge_size, max_size=a.max_edge_size,
          min_rating=a.min_rating_for_edge)
    return 0


if __name__ == "__main__":
    sys.exit(main())
