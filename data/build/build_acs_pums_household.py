"""Build ACS-PUMS-HH: persons grouped into households (ACS PUMS 2023 1-year).

Source: U.S. Census Bureau, American Community Survey Public Use Microdata
Sample, 2023 1-year, Massachusetts (public domain). ``data/download.py
--dataset acs_pums_household`` fetches csv_pma.zip and csv_hma.zip.

Construction (paper, Table 3):
    nodes       person records (SERIALNO, SPORDER)
    units       households (SERIALNO); housing units only (TYPEHUGQ == 1)
    unit size   2 <= |household| <= 10
    cap         at most 80,000 persons, sampled by household with seed 2026
                (Massachusetts has fewer, so no sampling happens)

Audited attributes: sex (SEX), age_bucket (AGEP), race (RAC1P), disability (DIS).
Label: education level (SCHL buckets).
Features: one-hot ESR, COW, OCCP major group, RELSHIPP, PINCP decile; they
exclude the audited attributes and the label.

Usage:
    python data/build/build_acs_pums_household.py
    python data/build/build_acs_pums_household.py --raw-dir <dir> --out-dir <dir>

``--raw-dir`` holds either the extracted CSVs (``ma_p/psam_p25.csv`` and
``ma_h/psam_h25.csv``) or the two zip files.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import common

STATE_FIPS = {
    "al": "01", "ak": "02", "az": "04", "ar": "05", "ca": "06",
    "co": "08", "ct": "09", "de": "10", "dc": "11", "fl": "12",
    "ga": "13", "hi": "15", "id": "16", "il": "17", "in": "18",
    "ia": "19", "ks": "20", "ky": "21", "la": "22", "me": "23",
    "md": "24", "ma": "25", "mi": "26", "mn": "27", "ms": "28",
    "mo": "29", "mt": "30", "ne": "31", "nv": "32", "nh": "33",
    "nj": "34", "nm": "35", "ny": "36", "nc": "37", "nd": "38",
    "oh": "39", "ok": "40", "or": "41", "pa": "42", "ri": "44",
    "sc": "45", "sd": "46", "tn": "47", "tx": "48", "ut": "49",
    "vt": "50", "va": "51", "wa": "53", "wv": "54", "wi": "55",
    "wy": "56", "pr": "72",
}

AGE_BUCKETS = [(0, 17, "0-17"), (18, 29, "18-29"), (30, 44, "30-44"),
               (45, 64, "45-64"), (65, 200, "65+")]

RAC1P_LABEL = {1: "white", 2: "black", 3: "aian", 4: "aian", 5: "aian",
               6: "asian", 7: "nhpi", 8: "other", 9: "multi"}

SCHL_BUCKETS = [
    (1, 11, "lt_hs"),
    (12, 15, "hs_some"),
    (16, 17, "hs_grad_or_ged"),
    (18, 19, "some_college"),
    (20, 20, "associates"),
    (21, 21, "bachelors"),
    (22, 22, "masters"),
    (23, 24, "professional_doctorate"),
]

P_COLS = ["RT", "SERIALNO", "SPORDER", "STATE", "PUMA", "SEX", "AGEP", "RAC1P",
          "DIS", "SCHL", "ESR", "OCCP", "PINCP", "WAGP", "COW", "HHL", "RELSHIPP",
          "PWGTP"]
H_COLS = ["SERIALNO", "STATE", "TYPE", "TYPEHUGQ", "NP", "HHL", "HINCP", "TEN", "WGTP"]

# The paper files were written on Windows: CRLF CSVs, Windows zip header.
EOL = common.CRLF
ZIP_HOST = common.ZIP_HOST_WINDOWS


def _bucket_age(age):
    if pd.isna(age):
        return "missing"
    a = int(age)
    for lo, hi, lab in AGE_BUCKETS:
        if lo <= a <= hi:
            return lab
    return "missing"


def _bucket_schl(schl):
    if pd.isna(schl):
        return "missing"
    s = int(schl)
    for lo, hi, lab in SCHL_BUCKETS:
        if lo <= s <= hi:
            return lab
    return "other"


def _occp_super(occp):
    if pd.isna(occp) or int(occp) <= 0:
        return "none"
    s = str(int(occp)).zfill(4)
    return f"occ_{s[:2]}"


def _read_state_csv(raw_dir: Path, state: str, kind: str, cols: list[str]) -> pd.DataFrame:
    fips = STATE_FIPS[state]
    name = f"psam_{kind}{fips}.csv"
    extracted = raw_dir / f"{state}_{kind}" / name
    kwargs = dict(dtype={"SERIALNO": str}, usecols=lambda c: c in cols, low_memory=False)
    if extracted.exists():
        return pd.read_csv(extracted, **kwargs)
    zpath = raw_dir / f"csv_{kind}{state}.zip"
    common.require(zpath, "run: python data/download.py --dataset acs_pums_household")
    with zipfile.ZipFile(zpath) as z, z.open(name) as fh:
        return pd.read_csv(fh, **kwargs)


def build(raw_dir: Path, out_dir: Path, *, states: list[str], year: int, max_persons: int,
          min_edge_size: int, max_edge_size: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    person_frames, house_frames = [], []
    for st in states:
        st = st.lower()
        if st not in STATE_FIPS:
            raise SystemExit(f"unknown state abbreviation: {st}")
        p = _read_state_csv(raw_dir, st, "p", P_COLS)
        h = _read_state_csv(raw_dir, st, "h", H_COLS)
        p["state_code"] = st
        h["state_code"] = st
        person_frames.append(p)
        house_frames.append(h)
    persons = pd.concat(person_frames, ignore_index=True)
    households = pd.concat(house_frames, ignore_index=True)

    if "TYPEHUGQ" in households.columns:
        households = households[households["TYPEHUGQ"] == 1].copy()
    elif "TYPE" in households.columns:
        households = households[households["TYPE"] == 1].copy()

    keep_serials = set(households["SERIALNO"].astype(str))
    persons = persons[persons["SERIALNO"].astype(str).isin(keep_serials)].copy()

    sizes = persons.groupby("SERIALNO").size()
    sizes.name = "_size"
    persons = persons.merge(sizes, left_on="SERIALNO", right_index=True, how="left")
    persons = persons[(persons["_size"] >= min_edge_size)
                      & (persons["_size"] <= max_edge_size)].copy()
    households = households[households["SERIALNO"].isin(persons["SERIALNO"])].copy()

    rng = np.random.default_rng(seed)
    sampled_serials = None
    if len(persons) > max_persons:
        order = households["SERIALNO"].astype(str).to_numpy().copy()
        rng.shuffle(order)
        size_lookup = sizes.to_dict()
        cum = 0
        keep_list = []
        for s in order:
            cum += int(size_lookup.get(s, 0))
            keep_list.append(s)
            if cum >= max_persons:
                break
        sampled_serials = set(keep_list)
        persons = persons[persons["SERIALNO"].astype(str).isin(sampled_serials)].copy()
        households = households[households["SERIALNO"].astype(str).isin(sampled_serials)].copy()

    persons = persons.sort_values(["SERIALNO", "SPORDER"]).reset_index(drop=True)
    persons["node_id"] = np.arange(len(persons), dtype=np.int64)
    house_id_map = {s: i for i, s in enumerate(households["SERIALNO"].astype(str))}
    persons["hyperedge_id"] = persons["SERIALNO"].astype(str).map(house_id_map)
    n = len(persons)
    m = len(households)
    if n < 1000:
        raise SystemExit(f"only {n} persons left after filtering")

    sens = pd.DataFrame({"node_id": persons["node_id"]})
    sens["sex"] = persons["SEX"].map({1: "male", 2: "female"}).fillna("missing")
    sens["age_bucket"] = persons["AGEP"].apply(_bucket_age)
    sens["race"] = persons["RAC1P"].map(RAC1P_LABEL).fillna("other")
    sens["disability"] = persons["DIS"].map({1: "yes", 2: "no"}).fillna("missing")

    label_field = "education_bucket"
    label_series = persons["SCHL"].apply(_bucket_schl)
    labels = pd.DataFrame({"node_id": persons["node_id"], "label": label_series.values})

    persons["OCCP_super"] = persons["OCCP"].apply(_occp_super)
    pincp = persons["PINCP"].fillna(0.0).astype(float)
    qs = np.unique(np.quantile(pincp, np.linspace(0, 1, 11)))
    persons["PINCP_decile"] = pd.cut(pincp, bins=qs, include_lowest=True, labels=False,
                                     duplicates="drop").fillna(0).astype(int).astype(str)

    # Same candidate list as the paper build; SCHL is the label and HHL is a
    # household-file column, so both are skipped.
    feature_blocks, feature_cols_used = [], []
    for col in ["SCHL", "ESR", "HHL", "COW", "OCCP_super", "RELSHIPP", "PINCP_decile"]:
        if col not in persons.columns or col == "SCHL":
            continue
        ser = persons[col].astype(str)
        codes, vocab = pd.factorize(ser.fillna("__missing__"))
        valid = codes >= 0
        rows = np.arange(len(codes))[valid]
        cols = codes[valid]
        block = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                              shape=(len(codes), max(len(vocab), 1)))
        feature_blocks.append(block)
        feature_cols_used.append({"name": col, "vocab_size": int(len(vocab))})
    X = sp.hstack(feature_blocks).tocsr()

    rows = persons["hyperedge_id"].to_numpy(dtype=np.int64)
    cols = persons["node_id"].to_numpy(dtype=np.int64)
    incid = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(m, n))

    common.save_npz(out_dir / "incidence.npz", incid, zip_host=ZIP_HOST)
    common.save_npz(out_dir / "node_features.npz", X, zip_host=ZIP_HOST)
    common.write_csv(sens, out_dir / "sensitive_attrs.csv", eol=EOL)
    common.write_csv(labels, out_dir / "labels.csv", eol=EOL)
    common.write_csv(pd.DataFrame({"node_id": persons["node_id"],
                                   "SERIALNO": persons["SERIALNO"].astype(str),
                                   "SPORDER": persons["SPORDER"].astype(int),
                                   "state_code": persons["state_code"]}),
                     out_dir / "node_mapping.csv", eol=EOL)
    common.write_csv(pd.DataFrame({"hyperedge_id": np.arange(m, dtype=np.int64),
                                   "SERIALNO": households["SERIALNO"].astype(str).to_numpy(),
                                   "state_code": households["state_code"].to_numpy()}),
                     out_dir / "hyperedge_mapping.csv", eol=EOL)

    edge_sizes = np.diff(incid.indptr)
    diag = common.basic_diagnostics(incid, sens, labels)
    diag["sampled"] = sampled_serials is not None
    metadata = {
        "dataset_name": "acs_pums_household",
        "paper_name": "ACS-PUMS-HH",
        "source": ("U.S. Census Bureau, American Community Survey Public Use Microdata "
                   f"Sample (ACS PUMS), {year} 1-Year"),
        "source_url_pattern": ("https://www2.census.gov/programs-surveys/acs/data/pums/"
                               f"{year}/1-Year/csv_{{p,h}}<state>.zip"),
        "license": "Public domain (U.S. federal government work).",
        "states_used": [s.lower() for s in states],
        "year": int(year),
        "construction_date": common.utc_now(),
        "random_seed": int(seed),
        "node_count": int(n),
        "hyperedge_count": int(m),
        "volume": int(incid.nnz),
        "feature_dim": int(X.shape[1]),
        "sensitive_attributes": [c for c in sens.columns if c != "node_id"],
        "label_source": label_field,
        "hyperedge_rule": "one unit per household (SERIALNO); members are its persons",
        "filters": {
            "household_type": "TYPEHUGQ==1 (housing unit; no group quarters)",
            "min_edge_size": int(min_edge_size),
            "max_edge_size": int(max_edge_size),
            "max_persons": int(max_persons),
            "household_level_sampling_seed": int(seed),
            "feature_excludes_sensitive_attrs": True,
            "feature_excludes_label_field": True,
        },
        "min_hyperedge_size": int(edge_sizes.min()),
        "max_hyperedge_size": int(edge_sizes.max()),
        "feature_cols_used": feature_cols_used,
    }
    readme = "\n".join([
        "# acs_pums_household (ACS-PUMS-HH)",
        "",
        f"States: {', '.join(metadata['states_used'])} ({year} 1-year ACS PUMS).",
        f"Nodes (persons): {n}. Units (households): {m}. Memberships: {incid.nnz}.",
        "Units are housing-unit households (TYPEHUGQ = 1) with 2 to 10 members.",
        "Audited attributes: sex, age_bucket, race, disability. Label: education level (SCHL buckets).",
        "Features: one-hot " + ", ".join(f["name"] for f in feature_cols_used)
        + "; they exclude the audited attributes and the label.",
        "",
        "Source: U.S. Census Bureau, ACS PUMS. Public domain (U.S. federal government work).",
        "",
    ])
    common.finish(out_dir, metadata, readme, diag)
    print(f"acs_pums_household: nodes={n} units={m} memberships={incid.nnz} "
          f"feature_dim={X.shape[1]} -> {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", type=Path, default=common.RAW_DIR / "acs_pums")
    ap.add_argument("--out-dir", type=Path, default=common.DATA_DIR / "acs_pums_household")
    ap.add_argument("--states", nargs="+", default=["ma"])
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--max-persons", type=int, default=80000)
    ap.add_argument("--min-edge-size", type=int, default=2)
    ap.add_argument("--max-edge-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=common.RANDOM_SEED)
    a = ap.parse_args()
    build(a.raw_dir, a.out_dir, states=a.states, year=a.year, max_persons=a.max_persons,
          min_edge_size=a.min_edge_size, max_edge_size=a.max_edge_size, seed=a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
