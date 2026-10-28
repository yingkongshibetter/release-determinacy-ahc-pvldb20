"""Build OA-DB: OpenAlex co-author units from the database venues.

Input: a JSONL cache of OpenAlex works (one work per line) whose primary
location is one of the six SIGMOD/VLDB/ICDE/PODS source records listed in
``openalex_api.DB_SOURCE_IDS``, created by
``python data/download.py --dataset openalex_db [--mailto ...]``.

The processed dataset used in the paper ships in ``data/openalex_db/``
(OpenAlex is CC0). This script documents and reruns its construction; a new
API fetch returns a newer snapshot, so its output will not match the shipped
files.

Construction (paper, Table 3):
 1. one unit per work; members are its authors with an OpenAlex id
 2. keep works with 3 to 50 author entries
 3. keep authors with >= 2 such works and a known country and institution type
 4. a unit keeps its surviving distinct authors; keep it if 3 to 50 remain;
    authors left without a unit are dropped
Audited attributes (per author): country and institution_type (most frequent
over the author's authorships; the first institution that has the field),
career_stage (earliest cached publication year: senior before 2010, mid
2010-2017, early from 2018). Label: most frequent primary-topic field over the
author's kept units. Features: OpenAlex topic counts, L1-normalised.

Usage:
    python data/build/build_openalex_db.py [--cache FILE] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import common
import openalex_api

EOL = common.LF                     # the paper files were written on Linux
ZIP_HOST = common.ZIP_HOST_UNIX


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _author_country(authorship: dict) -> str | None:
    for inst in authorship.get("institutions") or []:
        cc = inst.get("country_code")
        if cc:
            return str(cc)
    cc = authorship.get("country_code")
    return str(cc) if cc else None


def _author_inst_type(authorship: dict) -> str | None:
    for inst in authorship.get("institutions") or []:
        t = inst.get("type")
        if t:
            return str(t)
    return None


def _career_bucket(year: int | None) -> str | None:
    if year is None:
        return None
    if year < 2010:
        return "senior"
    if year < 2018:
        return "mid"
    return "early"


def _primary_field(work: dict) -> str | None:
    name = ((work.get("primary_topic") or {}).get("field") or {}).get("display_name")
    return str(name) if name else None


def build(cache: Path, out_dir: Path, *, min_size: int, max_size: int,
          min_author_works: int) -> None:
    common.require(cache, "run: python data/download.py --dataset openalex_db")
    work_authors, work_ids, work_field = [], [], []
    author_country: dict[str, Counter] = {}
    author_inst: dict[str, Counter] = {}
    author_year_min: dict[str, int] = {}
    author_topics: dict[str, Counter] = {}
    works_seen = works_kept = 0
    for work in _read_jsonl(cache):
        works_seen += 1
        wid = work.get("id")
        if not wid:
            continue
        ids = []
        for a in work.get("authorships") or []:
            aid = (a.get("author") or {}).get("id")
            if not aid:
                continue
            ids.append(str(aid))
            cc, it = _author_country(a), _author_inst_type(a)
            if cc:
                author_country.setdefault(aid, Counter())[cc] += 1
            if it:
                author_inst.setdefault(aid, Counter())[it] += 1
            year = work.get("publication_year")
            if year is not None:
                cur = author_year_min.get(aid)
                if cur is None or int(year) < cur:
                    author_year_min[aid] = int(year)
            for t in work.get("topics") or []:
                tid = t.get("id")
                if tid:
                    author_topics.setdefault(aid, Counter())[str(tid)] += 1
        if not (min_size <= len(ids) <= max_size):
            continue
        works_kept += 1
        work_authors.append(ids)
        work_ids.append(str(wid))
        work_field.append(_primary_field(work))
    if works_kept == 0:
        raise SystemExit("no works in the size window; check the cache")

    paper_count: Counter = Counter()
    for ids in work_authors:
        for a in ids:
            paper_count[a] += 1
    authors = sorted(a for a, c in paper_count.items() if c >= min_author_works)
    authors = [a for a in authors if a in author_country and a in author_inst]
    a2n = {a: i for i, a in enumerate(authors)}
    if not authors:
        raise SystemExit("no authors left after the attribute filter")

    units, unit_ids, raw_sizes, kept_sizes, unit_field = [], [], [], [], []
    for ids, ext, field in zip(work_authors, work_ids, work_field):
        kept = sorted({a2n[a] for a in ids if a in a2n})
        if not (min_size <= len(kept) <= max_size):
            continue
        units.append(kept)
        unit_ids.append(ext)
        raw_sizes.append(len(ids))
        kept_sizes.append(len(kept))
        unit_field.append(field)
    if not units:
        raise SystemExit("no units left after the author filter")

    used = sorted({v for u in units for v in u})
    if len(used) < len(authors):
        old_to_new = {old: new for new, old in enumerate(used)}
        units = [sorted(old_to_new[v] for v in u) for u in units]
        authors = [authors[v] for v in used]
        a2n = {a: i for i, a in enumerate(authors)}
    n_nodes, n_units = len(authors), len(units)

    rows = [e for e, u in enumerate(units) for _ in u]
    cols = [v for u in units for v in u]
    incidence = sp.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)),
                              shape=(n_units, n_nodes)).astype(np.int32, copy=False)

    topic_index: dict[str, int] = {}
    f_rows, f_cols, f_vals = [], [], []
    for a, node in a2n.items():
        for tid, cnt in author_topics.get(a, {}).items():
            if tid not in topic_index:
                topic_index[tid] = len(topic_index)
            f_rows.append(node)
            f_cols.append(topic_index[tid])
            f_vals.append(float(cnt))
    features = sp.csr_matrix((np.asarray(f_vals, dtype=np.float32),
                              (np.asarray(f_rows), np.asarray(f_cols))),
                             shape=(n_nodes, max(len(topic_index), 1)))
    row_sums = np.asarray(features.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    features = features.multiply((1.0 / row_sums)[:, None]).tocsr().astype(np.float32)

    sens = pd.DataFrame(
        [(node, author_country[a].most_common(1)[0][0], author_inst[a].most_common(1)[0][0],
          _career_bucket(author_year_min.get(a))) for a, node in a2n.items()],
        columns=["node_id", "country", "institution_type", "career_stage"],
    ).sort_values("node_id").reset_index(drop=True)

    field_counts: dict[int, Counter] = {}
    for members, field in zip(units, unit_field):
        if field:
            for v in members:
                field_counts.setdefault(v, Counter())[field] += 1
    labels = pd.DataFrame(
        [(v, field_counts[v].most_common(1)[0][0] if v in field_counts else "unknown")
         for v in range(n_nodes)], columns=["node_id", "label"])
    node_mapping = pd.DataFrame({"node_id": np.arange(n_nodes, dtype=np.int64),
                                 "external_id": authors})
    hyperedge_mapping = pd.DataFrame({
        "hyperedge_id": np.arange(n_units, dtype=np.int64), "external_id": unit_ids,
        "source_type": "openalex_work", "raw_size": raw_sizes, "kept_size": kept_sizes,
        "construction_rule": "one_hyperedge_per_work_db_venue"})

    out_dir.mkdir(parents=True, exist_ok=True)
    common.save_npz(out_dir / "incidence.npz", incidence, zip_host=ZIP_HOST)
    common.save_npz(out_dir / "node_features.npz", features, zip_host=ZIP_HOST)
    common.write_csv(sens, out_dir / "sensitive_attrs.csv", eol=EOL)
    common.write_csv(labels, out_dir / "labels.csv", eol=EOL)
    common.write_csv(node_mapping, out_dir / "node_mapping.csv", eol=EOL)
    common.write_csv(hyperedge_mapping, out_dir / "hyperedge_mapping.csv", eol=EOL)

    metadata = {
        "dataset_name": "openalex_db",
        "paper_name": "OA-DB",
        "raw_data_sources": [{"name": "OpenAlex Works (DB-venue filter)",
                              "url": openalex_api.API, "filter": openalex_api.DB_FILTER,
                              "select": openalex_api.DB_SELECT,
                              "source_ids_used": openalex_api.DB_SOURCE_IDS}],
        "construction_date": common.utc_now(),
        "random_seed": common.RANDOM_SEED,
        "node_count": n_nodes,
        "hyperedge_count": n_units,
        "volume": int(incidence.nnz),
        "feature_dim": int(features.shape[1]),
        "sensitive_attributes": ["country", "institution_type", "career_stage"],
        "label_source": "OpenAlex primary_topic.field.display_name",
        "hyperedge_rule": "one hyperedge per OpenAlex work; members are its authors",
        "filters": {"min_size": min_size, "max_size": max_size,
                    "min_author_works": min_author_works,
                    "drop_authors_missing_country": True,
                    "drop_authors_missing_inst_type": True},
        "min_hyperedge_size": min_size,
        "max_hyperedge_size": max_size,
        "openalex_works_seen": works_seen,
        "openalex_works_kept_after_size_filter": works_kept,
    }
    readme = (
        "# openalex_db (OA-DB)\n\n"
        "Units are OpenAlex works from the SIGMOD, VLDB, ICDE and PODS venues with 3 to 50 "
        "authors; members are their authors.\n"
        f"Nodes: {n_nodes}. Units: {n_units}. Memberships: {incidence.nnz}.\n"
        "Audited attributes: country, institution_type, career_stage. "
        "Label: most frequent primary-topic field.\n"
        "Features: OpenAlex topic counts, L1-normalised.\n\n"
        "Source: OpenAlex (CC0).\n"
    )
    common.finish(out_dir, metadata, readme,
                  common.basic_diagnostics(incidence, sens, labels))
    print(f"openalex_db: nodes={n_nodes} units={n_units} memberships={incidence.nnz} -> {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path,
                    default=common.RAW_DIR / "openalex" / "openalex_db_works.jsonl")
    ap.add_argument("--out-dir", type=Path, default=common.DATA_DIR / "openalex_db")
    ap.add_argument("--min-edge-size", type=int, default=3)
    ap.add_argument("--max-edge-size", type=int, default=50)
    ap.add_argument("--min-author-works", type=int, default=2)
    a = ap.parse_args()
    build(a.cache, a.out_dir, min_size=a.min_edge_size, max_size=a.max_edge_size,
          min_author_works=a.min_author_works)
    return 0


if __name__ == "__main__":
    sys.exit(main())
