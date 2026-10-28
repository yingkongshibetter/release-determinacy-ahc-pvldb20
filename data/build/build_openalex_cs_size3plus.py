"""Build OA-CS-3+: OpenAlex computer-science co-author units of size 3 to 25.

Input: a JSONL cache of OpenAlex works (one work per line) for the filter
``concepts.id:C41008148,publication_year:2018-2024``, created by
``python data/download.py --dataset openalex_cs_size3plus [--mailto ...]``
(first 30,000 works returned by the API).

The processed dataset used in the paper ships in ``data/openalex_cs_size3plus/``
(OpenAlex is CC0). This script documents and reruns its construction; a new
API fetch returns different works, so its output will not match the shipped
files.

Construction, in the order the paper's pipeline applied it:
 1. one unit per work; members are the distinct authors with an OpenAlex id
 2. keep works with 2 to 25 distinct authors
 3. keep authors with >= 2 such works and a known country, institution type,
    publication year and primary-topic field; re-apply the size window
 4. repeatedly drop authors that appear in fewer than 2 surviving works, and
    works that leave the size window, until stable
 5. keep units with >= 3 members and drop authors left without a unit
Audited attributes (per author, from the cached works): country and
institution_type (most frequent affiliation; the first institution of each
authorship), career_stage (earliest observed year: early if 2022-2024, mid if
2020-2021, senior otherwise). Label: most frequent primary-topic field.
Features: OpenAlex topic counts, L1-normalised.

Usage:
    python data/build/build_openalex_cs_size3plus.py [--cache FILE] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

import common
import openalex_api

EOL = common.LF                     # the paper files were written on Linux
ZIP_HOST = common.ZIP_HOST_UNIX


def get_id(url: str | None) -> str | None:
    return url.rsplit("/", 1)[-1] if url else None


def parse_works(cache_file: Path) -> list[dict]:
    works = []
    with cache_file.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            w = json.loads(line)
            authors = []
            for au in w.get("authorships") or []:
                a = au.get("author") or {}
                aid = get_id(a.get("id"))
                if not aid:
                    continue
                inst_list = au.get("institutions") or []
                if inst_list:
                    country = inst_list[0].get("country_code")
                    inst_type = inst_list[0].get("type")
                else:
                    country = inst_type = None
                if not country:
                    cc = au.get("countries") or []
                    if cc:
                        country = cc[0]
                authors.append({"author_id": aid, "country": country,
                                "institution_type": inst_type})
            if not authors:
                continue
            pt = w.get("primary_topic") or {}
            field = pt.get("field") or {}
            topics = [get_id(t.get("id")) for t in (w.get("topics") or [])]
            works.append({
                "work_id": get_id(w.get("id")),
                "year": w.get("publication_year"),
                "primary_field_id": get_id(field.get("id")),
                "primary_field_name": field.get("display_name"),
                "topics": [t for t in topics if t],
                "authors": authors,
            })
    return works


def aggregate_authors(works: list[dict]) -> dict[str, dict]:
    authors: dict[str, dict] = defaultdict(lambda: {
        "country_counts": Counter(), "inst_type_counts": Counter(), "years": [],
        "topic_counts": Counter(), "field_counts": Counter(), "n_works": 0})
    for w in works:
        for a in w["authors"]:
            e = authors[a["author_id"]]
            if a["country"]:
                e["country_counts"][a["country"]] += 1
            if a["institution_type"]:
                e["inst_type_counts"][a["institution_type"]] += 1
            if w["year"]:
                e["years"].append(int(w["year"]))
            for tid in w["topics"]:
                e["topic_counts"][tid] += 1
            if w["primary_field_id"]:
                e["field_counts"][w["primary_field_id"]] += 1
            e["n_works"] += 1
    return authors


def career_bucket(first_year: int, year_end: int) -> str:
    delta = year_end - first_year
    if delta <= 2:
        return "early"
    if delta <= 4:
        return "mid"
    return "senior"


def _n_distinct(w: dict) -> int:
    return len({a["author_id"] for a in w["authors"]})


def filter_works(works, *, min_size, max_size, min_author_works, min_node_degree):
    after_size = [w for w in works if min_size <= _n_distinct(w) <= max_size]
    authors = aggregate_authors(after_size)
    kept = {aid: e for aid, e in authors.items()
            if e["n_works"] >= min_author_works and e["country_counts"]
            and e["inst_type_counts"] and e["years"] and e["field_counts"]}
    final_works = []
    for w in after_size:
        ka = [a for a in w["authors"] if a["author_id"] in kept]
        if min_size <= len({a["author_id"] for a in ka}) <= max_size:
            final_works.append(dict(w, authors=ka))
    appearing = {a["author_id"] for w in final_works for a in w["authors"]}
    final_authors = {aid: kept[aid] for aid in appearing}
    iters = 0
    while min_node_degree > 1:
        iters += 1
        deg: Counter = Counter()
        for w in final_works:
            for a in w["authors"]:
                deg[a["author_id"]] += 1
        low = {aid for aid, c in deg.items() if c < min_node_degree}
        if not low:
            break
        new_works = []
        for w in final_works:
            ka = [a for a in w["authors"] if a["author_id"] not in low]
            if min_size <= len({a["author_id"] for a in ka}) <= max_size:
                new_works.append(dict(w, authors=ka))
        for aid in low:
            final_authors.pop(aid, None)
        appearing = {a["author_id"] for w in new_works for a in w["authors"]}
        for aid in list(final_authors):
            if aid not in appearing:
                final_authors.pop(aid, None)
        final_works = new_works
        if iters > 50:
            break
    return final_works, final_authors


def assemble_base(final_works, final_authors, *, year_end):
    """The size>=2 dataset, as the paper pipeline stored it before the size-3 cut."""
    authors = sorted(final_authors)
    a2n = {aid: i for i, aid in enumerate(authors)}
    work_ids = sorted({w["work_id"] for w in final_works})
    w2e = {wid: i for i, wid in enumerate(work_ids)}
    by_id = {w["work_id"]: w for w in final_works}
    rows, cols, sizes = [], [], []
    for wid in work_ids:
        members = sorted({a2n[a["author_id"]] for a in by_id[wid]["authors"]})
        rows.extend([w2e[wid]] * len(members))
        cols.extend(members)
        sizes.append(len(members))
    H = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                      shape=(len(work_ids), len(authors)))

    topics = sorted({t for e in final_authors.values() for t in e["topic_counts"]})
    t2c = {t: j for j, t in enumerate(topics)}
    X = sp.lil_matrix((len(authors), len(topics)), dtype=np.float32)
    for aid, e in final_authors.items():
        for tid, c in e["topic_counts"].items():
            X[a2n[aid], t2c[tid]] = float(c)
    X = X.tocsr()
    rs = np.asarray(X.sum(axis=1)).ravel()
    safe = rs.copy()
    safe[safe == 0] = 1.0
    X = (sp.diags(1.0 / safe) @ X).tocsr()

    field_name: dict[str, str] = {}
    for w in final_works:
        if w["primary_field_id"] and w["primary_field_name"]:
            field_name.setdefault(w["primary_field_id"], w["primary_field_name"])
    sens_rows, lab_rows = [], []
    for aid in authors:
        e = final_authors[aid]
        sens_rows.append((a2n[aid], aid, e["country_counts"].most_common(1)[0][0],
                          e["inst_type_counts"].most_common(1)[0][0],
                          career_bucket(min(e["years"]), year_end)))
        fid = e["field_counts"].most_common(1)[0][0]
        lab_rows.append((a2n[aid], aid, field_name.get(fid, fid)))
    sens = pd.DataFrame(sens_rows, columns=["node_id", "author_id", "country",
                                            "institution_type", "career_stage"])
    labels = pd.DataFrame(lab_rows, columns=["node_id", "author_id", "dominant_label"])
    nodes = pd.DataFrame({"node_id": np.arange(len(authors)), "author_id": authors})
    edges = pd.DataFrame({"hyperedge_id": np.arange(len(work_ids)), "work_id": work_ids,
                          "size": sizes})
    return H, X, sens, labels, nodes, edges


def to_canonical(H, X, sens, labels, nodes, edges):
    """Rename columns to the shared layout; CSV round trips match the paper pipeline."""
    incidence = H.tocsr().astype(np.int32)
    features = X.tocsr().astype(np.float32)
    sens = common.csv_roundtrip(sens)
    sens = sens[["node_id"] + [c for c in sens.columns
                               if c not in ("node_id", "author_id", "display_name")]].copy()
    labels = common.csv_roundtrip(labels).rename(columns={"dominant_label": "label"})
    labels = labels[["node_id", "label"]].copy()
    nodes = common.csv_roundtrip(nodes).rename(columns={"author_id": "external_id"})
    nodes = nodes[["node_id", "external_id"]].copy()
    edges = common.csv_roundtrip(edges)
    sizes = np.diff(incidence.indptr).astype(np.int64)
    hyperedges = pd.DataFrame({
        "hyperedge_id": np.arange(incidence.shape[0], dtype=np.int64),
        "external_id": edges["work_id"].astype(str).tolist(),
        "source_type": "openalex_work",
        "raw_size": sizes,
        "kept_size": sizes,
        "construction_rule": "one_hyperedge_per_work_cs_concept",
    })
    # The intermediate dataset was saved to CSV and read back before the size-3 cut.
    return (incidence, features, common.csv_roundtrip(sens.sort_values("node_id")),
            common.csv_roundtrip(labels.sort_values("node_id")),
            common.csv_roundtrip(nodes.sort_values("node_id")),
            common.csv_roundtrip(hyperedges.sort_values("hyperedge_id")))


def size_cut(incidence, features, sens, labels, nodes, hyperedges, *, min_size):
    sizes = np.diff(incidence.indptr)
    kept = np.flatnonzero(sizes >= int(min_size))
    sub = incidence[kept]
    used = np.flatnonzero(np.asarray(sub.sum(axis=0)).ravel() > 0)
    sub = sub[:, used]
    feat = features[used]
    new_ids = np.arange(used.shape[0], dtype=np.int64)

    def take(df, cols):
        out = df.set_index("node_id").loc[used].reset_index(drop=True)
        out.insert(0, "node_id", new_ids)
        return out[cols] if cols else out

    sens = take(sens, None)
    labels = take(labels, ["node_id", "label"])
    nodes = take(nodes, ["node_id", "external_id"])
    hyperedges = hyperedges.iloc[kept].reset_index(drop=True)
    hyperedges["hyperedge_id"] = np.arange(len(hyperedges), dtype=np.int64)
    return sub, feat, sens, labels, nodes, hyperedges


def build(cache: Path, out_dir: Path, *, min_size: int, max_size: int, min_author_works: int,
          min_node_degree: int, final_min_size: int, year_end: int) -> None:
    common.require(cache, "run: python data/download.py --dataset openalex_cs_size3plus")
    works = parse_works(cache)
    final_works, final_authors = filter_works(works, min_size=min_size, max_size=max_size,
                                              min_author_works=min_author_works,
                                              min_node_degree=min_node_degree)
    base = assemble_base(final_works, final_authors, year_end=year_end)
    print(f"size>=2 dataset: nodes={base[0].shape[1]} units={base[0].shape[0]}")
    H, X, sens, labels, nodes, hyperedges = size_cut(*to_canonical(*base),
                                                     min_size=final_min_size)
    H = H.astype(np.int32, copy=False)
    X = X.astype(np.float32, copy=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    common.save_npz(out_dir / "incidence.npz", H, zip_host=ZIP_HOST)
    common.save_npz(out_dir / "node_features.npz", X, zip_host=ZIP_HOST)
    common.write_csv(sens.sort_values("node_id"), out_dir / "sensitive_attrs.csv", eol=EOL)
    common.write_csv(labels.sort_values("node_id"), out_dir / "labels.csv", eol=EOL)
    common.write_csv(nodes.sort_values("node_id"), out_dir / "node_mapping.csv", eol=EOL)
    common.write_csv(hyperedges.sort_values("hyperedge_id"), out_dir / "hyperedge_mapping.csv",
                     eol=EOL)

    sizes = np.diff(H.indptr)
    metadata = {
        "dataset_name": "openalex_cs_size3plus",
        "paper_name": "OA-CS-3+",
        "raw_data_sources": [{"name": "OpenAlex Works (CS-concept filter)",
                              "url": openalex_api.API,
                              "filter": openalex_api.CS_FILTER,
                              "select": openalex_api.CS_SELECT,
                              "works_in_cache": len(works)}],
        "construction_date": common.utc_now(),
        "random_seed": common.RANDOM_SEED,
        "node_count": int(H.shape[1]),
        "hyperedge_count": int(H.shape[0]),
        "volume": int(H.nnz),
        "feature_dim": int(X.shape[1]),
        "sensitive_attributes": ["country", "institution_type", "career_stage"],
        "label_source": "OpenAlex primary_topic.field.display_name",
        "hyperedge_rule": "one hyperedge per OpenAlex CS work; members are its authors",
        "filters": {"concept_id": openalex_api.CS_CONCEPT, "year_range": "2018-2024",
                    "min_size": min_size, "max_size": max_size,
                    "min_author_works": min_author_works,
                    "min_node_degree_closure": min_node_degree,
                    "final_min_size": final_min_size},
        "min_hyperedge_size": int(sizes.min()),
        "max_hyperedge_size": int(sizes.max()),
    }
    readme = (
        "# openalex_cs_size3plus (OA-CS-3+)\n\n"
        "Units are OpenAlex computer-science works (2018-2024) with 3 to 25 authors; "
        "members are their authors.\n"
        f"Nodes: {H.shape[1]}. Units: {H.shape[0]}. Memberships: {H.nnz}.\n"
        "Audited attributes: country, institution_type, career_stage. "
        "Label: most frequent primary-topic field.\n"
        "Features: OpenAlex topic counts, L1-normalised.\n\n"
        "Source: OpenAlex (CC0).\n"
    )
    common.finish(out_dir, metadata, readme, common.basic_diagnostics(H, sens, labels))
    print(f"openalex_cs_size3plus: nodes={H.shape[1]} units={H.shape[0]} "
          f"memberships={H.nnz} -> {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=common.RAW_DIR / "openalex" / "cs_works.jsonl")
    ap.add_argument("--out-dir", type=Path, default=common.DATA_DIR / "openalex_cs_size3plus")
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--max-size", type=int, default=25)
    ap.add_argument("--min-author-works", type=int, default=2)
    ap.add_argument("--min-node-degree", type=int, default=2)
    ap.add_argument("--final-min-size", type=int, default=3)
    ap.add_argument("--year-end", type=int, default=2024)
    a = ap.parse_args()
    build(a.cache, a.out_dir, min_size=a.min_size, max_size=a.max_size,
          min_author_works=a.min_author_works, min_node_degree=a.min_node_degree,
          final_min_size=a.final_min_size, year_end=a.year_end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
