"""Build Pokec: Pokec users grouped by a shared hobby or interest token.

Input: SNAP ``soc-pokec-profiles.txt.gz`` (or the decompressed .txt) in
``data/raw/pokec/``, fetched by ``python data/download.py --dataset pokec_hobby``.
The friendship graph is not used.

Construction (paper, Table 3):
    profiles    the first 200,000 lines of soc-pokec-profiles.txt
    tokens      the hobby/interest fields in HOBBY_FIELDS are split on , ; | and
                newlines, lower-cased and whitespace-normalised; null and generic
                tokens ("ostatne", "ine", "vsetko", ...) are dropped
    units       one unit per token, members are the users who declare it;
                3 <= size <= 50
    nodes       users that belong to at least one unit
Audited attributes: gender, AGE (bucketed), region_kraj (the region prefix
before the first comma; the full region string is kept as region_full).
Label: education level, the highest level named in
completed_level_of_education (EDUCATION_LEVELS).
Features: bag of hobby tokens used by >= 5 of the 200,000 users, excluding
any token equal to an audited-attribute or label value; L1-normalised.

Usage:
    python data/build/build_pokec_hobby.py [--raw-dir DIR] [--out-dir DIR]
Runtime about 2 minutes.
"""
from __future__ import annotations

import argparse
import gzip
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import common

# Positional columns of soc-pokec-profiles.txt (SNAP schema, 59 fields).
COLUMNS = [
    "user_id", "public", "completion_percentage", "gender", "region", "last_login",
    "registration", "AGE", "body", "I_am_working_in_field", "spoken_languages", "hobbies",
    "I_most_enjoy_good_food", "pets", "body_type", "my_eyesight", "eye_color", "hair_color",
    "hair_type", "completed_level_of_education", "favourite_color", "relation_to_smoking",
    "relation_to_alcohol", "sign_in_zodiac", "on_pokec_i_am_looking_for", "love_is_for_me",
    "relation_to_casual_sex", "my_partner_should_be", "marital_status", "children",
    "relation_to_children", "I_like_movies", "I_like_watching_movie", "I_like_music",
    "I_mostly_like_listening_to_music", "the_idea_of_good_evening",
    "I_like_specialties_from_kitchen", "fun", "I_am_going_to_concerts", "my_active_sports",
    "my_passive_sports", "profession", "I_like_books", "life_style", "music", "cars",
    "politics", "relationships", "art_culture", "hobbies_other", "science_technologies",
    "computers_internet", "education", "sport", "movies", "travelling", "health",
    "companies_brands", "more",
]
HOBBY_FIELDS = ["hobbies", "hobbies_other", "my_active_sports", "my_passive_sports",
                "I_like_music", "I_like_books", "I_like_movies", "I_am_working_in_field",
                "sport"]
SENSITIVE_FIELDS = ["gender", "AGE", "region"]
LABEL_FIELD = "completed_level_of_education"
USER_ID = "user_id"

_NULL_TOKENS = {"null", "none", "", "?", "n/a"}
_GENERIC_TOKENS = {"ostatne", "iné", "ine", "vsetko", "všetko"}  # "other", "all"

# Education label: the first key found in the lower-cased field wins.
EDUCATION_LEVELS = [
    ("doktorske", "doctoral"),
    ("vysokoskolske", "tertiary_long"),
    ("vysokoskolske s bakalarskym titulom", "tertiary_short"),
    ("vyssie", "tertiary_short"),
    ("ucnovske", "vocational"),
    ("stredoskolske", "secondary"),
    ("zakladne", "basic"),
]

# Line endings and zip host of the paper files: built on Linux; labels.csv
# was rewritten later on Windows.
EOL_MAIN = common.LF
EOL_LABELS = common.CRLF
ZIP_HOST = common.ZIP_HOST_UNIX


def _normalise_token(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def _parse_hobby_field(raw) -> list[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    s = str(raw)
    if s.lower() in _NULL_TOKENS:
        return []
    out = []
    for p in re.split(r"[,;|\n]+", s):
        nt = _normalise_token(p)
        if nt and nt not in _NULL_TOKENS and nt not in _GENERIC_TOKENS:
            out.append(nt)
    return out


def _age_bucket(age_raw) -> str:
    try:
        age = int(float(age_raw))
    except (TypeError, ValueError):
        return "unknown"
    if age <= 0 or age > 120:
        return "unknown"
    if age < 18:
        return "<18"
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 60:
        return "45-59"
    return "60+"


def _kraj(text):
    if text is None or isinstance(text, float):
        return None
    s = str(text).strip().lower()
    if not s or s == "nan":
        return None
    return s.split(",", 1)[0].strip() or None


def _education(s: str) -> str:
    for key, value in EDUCATION_LEVELS:
        if key in s:
            return value
    return "other_or_missing"


def _open_profiles(raw_dir: Path):
    txt = raw_dir / "soc-pokec-profiles.txt"
    if txt.exists():
        return txt.open("r", encoding="utf-8", errors="replace")
    gz = common.require(raw_dir / "soc-pokec-profiles.txt.gz",
                        "run: python data/download.py --dataset pokec_hobby")
    return gzip.open(gz, "rt", encoding="utf-8", errors="replace")


def read_profiles(raw_dir: Path, max_rows: int) -> pd.DataFrame:
    needed = [USER_ID] + HOBBY_FIELDS + SENSITIVE_FIELDS + [LABEL_FIELD]
    rows = []
    with _open_profiles(raw_dir) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(COLUMNS):
                parts.extend([""] * (len(COLUMNS) - len(parts)))
            rec = dict(zip(COLUMNS, parts))
            rows.append({k: rec.get(k, "") for k in needed})
            if len(rows) >= max_rows:
                break
    if not rows:
        raise SystemExit("no profile rows read")
    return pd.DataFrame(rows)


def build(raw_dir: Path, out_dir: Path, *, max_rows: int, min_size: int, max_size: int,
          min_token_count: int) -> None:
    df = read_profiles(raw_dir, max_rows)

    tokens_per_user = []
    for _, row in df[HOBBY_FIELDS].iterrows():
        tokens: set[str] = set()
        for v in row.values:
            tokens.update(_parse_hobby_field(v))
        tokens_per_user.append(sorted(tokens))

    token_to_users: dict[str, list[int]] = defaultdict(list)
    for u, tokens in enumerate(tokens_per_user):
        for t in tokens:
            token_to_users[t].append(u)
    units, rejected = [], 0
    for token, members in token_to_users.items():
        if min_size <= len(members) <= max_size:
            units.append((token, sorted(set(members))))
        else:
            rejected += 1
    if not units:
        raise SystemExit("no hobby token in the size window")

    surviving = sorted({u for _, members in units for u in members})
    user_to_node = {u: i for i, u in enumerate(surviving)}
    n_nodes, n_units = len(surviving), len(units)
    rows = [e for e, (_, members) in enumerate(units) for _ in members]
    cols = [user_to_node[u] for _, members in units for u in members]
    incidence = sp.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)),
                              shape=(n_units, n_nodes)).astype(np.int32, copy=False)

    blacklist: set[str] = set()
    for col in SENSITIVE_FIELDS + [LABEL_FIELD]:
        blacklist.update(_normalise_token(str(v)) for v in df[col].dropna().unique())
    counts: Counter = Counter()
    for tokens in tokens_per_user:
        for t in tokens:
            if t not in blacklist:
                counts[t] += 1
    vocab = sorted(t for t, c in counts.items() if c >= min_token_count)
    token_index = {t: i for i, t in enumerate(vocab)}
    f_rows, f_cols = [], []
    for u, tokens in enumerate(tokens_per_user):
        if u not in user_to_node:
            continue
        for t in tokens:
            j = token_index.get(t)
            if j is not None:
                f_rows.append(user_to_node[u])
                f_cols.append(j)
    features = sp.csr_matrix((np.ones(len(f_rows), dtype=np.float32),
                              (np.asarray(f_rows), np.asarray(f_cols))),
                             shape=(n_nodes, max(len(token_index), 1)))
    row_sums = np.asarray(features.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    features = features.multiply((1.0 / row_sums)[:, None]).tocsr().astype(np.float32)

    # Audited attributes. The paper pipeline stored gender/AGE/region first and
    # derived region_kraj from the stored CSV, so the CSV round trip is kept.
    sens = {"node_id": list(range(n_nodes))}
    for sf in SENSITIVE_FIELDS:
        vals = df[sf].tolist()
        sens[sf] = [_age_bucket(vals[u]) if sf == "AGE"
                    else (str(vals[u]) if vals[u] not in ("", None) else None)
                    for u in surviving]
    sens = common.csv_roundtrip(pd.DataFrame(sens).sort_values("node_id"))
    sens["region_kraj"] = sens["region"].map(_kraj)
    sens = sens.rename(columns={"region": "region_full"})
    sens = sens[["node_id", "gender", "AGE", "region_kraj", "region_full"]]

    # Label: free-text education field, lower-cased, then collapsed to its
    # highest level (again through the stored CSV, as in the paper pipeline).
    raw_label = df[LABEL_FIELD].tolist()
    label_text = pd.DataFrame({
        "node_id": list(range(n_nodes)),
        "label": [str(raw_label[u]).strip().lower() if raw_label[u] not in ("", None)
                  else "unknown" for u in surviving]})
    label_text = common.csv_roundtrip(label_text)
    lowered = label_text["label"].astype(str).fillna("__missing__").str.lower()
    labels = pd.DataFrame({"node_id": np.arange(n_nodes), "label": lowered.map(_education).values})

    node_mapping = pd.DataFrame({"node_id": np.arange(n_nodes, dtype=np.int64),
                                 "external_id": df.iloc[surviving][USER_ID].tolist()})
    hyperedge_mapping = pd.DataFrame({
        "hyperedge_id": np.arange(n_units, dtype=np.int64),
        "external_id": [t for t, _ in units],
        "source_type": "hobby_token",
        "raw_size": [len(m) for _, m in units],
        "kept_size": [len(m) for _, m in units],
        "construction_rule": "users sharing a normalised hobby/interest token",
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    common.save_npz(out_dir / "incidence.npz", incidence, zip_host=ZIP_HOST)
    common.save_npz(out_dir / "node_features.npz", features, zip_host=ZIP_HOST)
    common.write_csv(sens, out_dir / "sensitive_attrs.csv", eol=EOL_MAIN)
    common.write_csv(labels, out_dir / "labels.csv", eol=EOL_LABELS)
    common.write_csv(node_mapping, out_dir / "node_mapping.csv", eol=EOL_MAIN)
    common.write_csv(hyperedge_mapping, out_dir / "hyperedge_mapping.csv", eol=EOL_MAIN)

    metadata = {
        "dataset_name": "pokec_hobby",
        "paper_name": "Pokec",
        "raw_data_sources": [{"name": "Pokec social network (SNAP)",
                              "url": "https://snap.stanford.edu/data/soc-Pokec.html"}],
        "construction_date": common.utc_now(),
        "random_seed": common.RANDOM_SEED,
        "node_count": n_nodes,
        "hyperedge_count": n_units,
        "volume": int(incidence.nnz),
        "feature_dim": int(features.shape[1]),
        "sensitive_attributes": ["gender", "AGE", "region_kraj"],
        "label_source": "highest education level named in completed_level_of_education",
        "hyperedge_rule": "users sharing a normalised hobby/interest token",
        "filters": {"min_size": min_size, "max_size": max_size, "max_nodes": max_rows,
                    "hobby_fields": HOBBY_FIELDS, "sensitive_fields": SENSITIVE_FIELDS,
                    "drop_friendship_graph_edges": True,
                    "vocab_min_token_count": min_token_count,
                    "region_kraj_extraction_rule": "lowercase(region.split(',')[0])",
                    "region_full_kept_as": "region_full (auxiliary, not audited)"},
        "min_hyperedge_size": min_size,
        "max_hyperedge_size": max_size,
        "removed_node_count": int(len(df) - n_nodes),
        "removed_hyperedge_count": int(rejected),
        "vocab_size": len(vocab),
        "education_levels": dict(EDUCATION_LEVELS),
    }
    readme = (
        "# pokec_hobby (Pokec)\n\n"
        "Units are users (first 200,000 SNAP profiles) who share one normalised hobby or "
        f"interest token, size in [{min_size}, {max_size}]. The friendship graph is not used.\n"
        f"Nodes: {n_nodes}. Units: {n_units}. Memberships: {incidence.nnz}.\n"
        "Audited attributes: gender, AGE, region_kraj (region_full is auxiliary). "
        "Label: education level.\n"
        "Features: bag of hobby tokens, L1-normalised.\n\n"
        "Built from the SNAP Pokec profiles (https://snap.stanford.edu/data/soc-Pokec.html); "
        "cite L. Takac and M. Zabovsky (2012).\n"
    )
    common.finish(out_dir, metadata, readme, common.basic_diagnostics(incidence, sens, labels))
    print(f"pokec_hobby: nodes={n_nodes} units={n_units} memberships={incidence.nnz} -> {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", type=Path, default=common.RAW_DIR / "pokec")
    ap.add_argument("--out-dir", type=Path, default=common.DATA_DIR / "pokec_hobby")
    ap.add_argument("--max-rows", type=int, default=200000)
    ap.add_argument("--min-edge-size", type=int, default=3)
    ap.add_argument("--max-edge-size", type=int, default=50)
    ap.add_argument("--min-token-count", type=int, default=5)
    a = ap.parse_args()
    build(a.raw_dir, a.out_dir, max_rows=a.max_rows, min_size=a.min_edge_size,
          max_size=a.max_edge_size, min_token_count=a.min_token_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
