"""Download the raw sources of the paper's datasets into data/raw/<source>/.

Examples:
    python data/download.py --dataset movielens_1m
    python data/download.py --dataset all            # ML-1M, Pokec, IMDb, ACS PUMS
    python data/download.py --dataset openalex_db --mailto YOUR_EMAIL

``all`` covers the file downloads. The OpenAlex queries must be named
explicitly: the processed OpenAlex datasets ship with the repository, and a
new query returns a newer snapshot than the one used in the paper.

Nothing here is redistributed; each source keeps its own licence (printed
before every download). data/raw/ is not part of the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "build"))
import openalex_api  # noqa: E402

RAW_DIR = HERE / "raw"
USER_AGENT = "release-determinacy-ahc-data/1.0"

LICENCES = {
    "movielens": (
        "MovieLens-1M (GroupLens Research, University of Minnesota). Research use under the "
        "GroupLens usage licence: no redistribution without permission, no commercial use, "
        "cite F. M. Harper and J. A. Konstan, ACM TiiS 5(4), 2015. "
        "https://files.grouplens.org/datasets/movielens/ml-1m-README.txt"),
    "pokec": (
        "Pokec social network profiles (SNAP, Stanford). Free for research use with "
        "citation of L. Takac and M. Zabovsky, 2012. https://snap.stanford.edu/data/soc-Pokec.html"),
    "imdb": (
        "IMDb non-commercial datasets. Personal and non-commercial use only, subject to the "
        "IMDb terms; do not redistribute. https://developer.imdb.com/non-commercial-datasets/"),
    "acs_pums": (
        "American Community Survey PUMS, U.S. Census Bureau. Public domain (U.S. federal "
        "government work); cite the Census Bureau as the source."),
    "openalex": (
        "OpenAlex (OurResearch). CC0 1.0. https://openalex.org/ ; API documentation "
        "https://docs.openalex.org/"),
}

# sha256 of the raw files used in the paper, where the source is a fixed release.
KNOWN_SHA256 = {
    "ml-1m.zip": "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
    "csv_pma.zip": "0e50dbb808f383b340d11ee55ef0271f9e6739d3410f50b7c7329c87e1b71654",
    "csv_hma.zip": "0c6e9324c91fe632984cef981b9bcf9fc11200ec4fd8da54d3fec083daf07660",
}

ML_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
POKEC_URLS = ["https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz",
              "https://snap.stanford.edu/data/soc-pokec-readme.txt"]
IMDB_URL = "https://datasets.imdbws.com/"
IMDB_FILES = ["title.basics.tsv.gz", "title.principals.tsv.gz", "name.basics.tsv.gz"]
ACS_URL = "https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/"
ACS_FIPS = {"ma": "25"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  exists: {dest.name}")
    else:
        print(f"  fetching {url}", flush=True)
        partial = dest.with_name(dest.name + ".partial")
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=300) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
        partial.replace(dest)
    digest = _sha256(dest)
    expected = KNOWN_SHA256.get(dest.name)
    note = ""
    if expected:
        note = " (matches the file used in the paper)" if digest == expected else \
            " (WARNING: differs from the file used in the paper)"
    print(f"  {dest.name}: {dest.stat().st_size:,} bytes, sha256 {digest}{note}")
    return {"url": url, "file": dest.name, "bytes": dest.stat().st_size, "sha256": digest}


def _extract(zpath: Path, members: dict[str, Path]) -> None:
    """Extract the named members only (no paths taken from the archive)."""
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
        for member, dest in members.items():
            if member not in names:
                raise SystemExit(f"{zpath.name} lacks {member}")
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out, length=1 << 20)


def _manifest(folder: Path, records: list[dict]) -> None:
    (folder / "download_manifest.json").write_text(json.dumps(
        {"downloaded_utc": datetime.now(timezone.utc).isoformat(), "files": records},
        indent=2), encoding="utf-8")


def get_movielens(raw: Path) -> None:
    folder = raw / "movielens"
    rec = [download(ML_URL, folder / "ml-1m.zip")]
    _extract(folder / "ml-1m.zip",
             {f"ml-1m/{n}": folder / "ml-1m" / n
              for n in ("users.dat", "movies.dat", "ratings.dat", "README")})
    _manifest(folder, rec)


def get_pokec(raw: Path) -> None:
    folder = raw / "pokec"
    _manifest(folder, [download(u, folder / u.rsplit("/", 1)[-1]) for u in POKEC_URLS])


def get_imdb(raw: Path) -> None:
    folder = raw / "imdb"
    print("  note: IMDb refreshes these files daily; the paper used the 2026-05-07 snapshot.")
    _manifest(folder, [download(IMDB_URL + f, folder / f) for f in IMDB_FILES])


def get_acs(raw: Path, states: list[str]) -> None:
    folder = raw / "acs_pums"
    rec = []
    for st in states:
        fips = ACS_FIPS.get(st)
        if fips is None:
            raise SystemExit("only the Massachusetts files (ma) are used in the paper; "
                             "add the state FIPS code to ACS_FIPS for other states")
        for kind in ("p", "h"):
            zname = f"csv_{kind}{st}.zip"
            rec.append(download(ACS_URL + zname, folder / zname))
            _extract(folder / zname,
                     {f"psam_{kind}{fips}.csv": folder / f"{st}_{kind}" / f"psam_{kind}{fips}.csv"})
    _manifest(folder, rec)


def get_openalex(raw: Path, which: str, mailto: str | None) -> None:
    folder = raw / "openalex"
    print("  note: OpenAlex changes over time; the paper used a May 2026 snapshot, which ships "
          "as processed data in data/openalex_db and data/openalex_cs_size3plus.")
    if which == "openalex_db":
        openalex_api.fetch_jsonl(folder / "openalex_db_works.jsonl",
                                 filter_value=openalex_api.DB_FILTER,
                                 select=openalex_api.DB_SELECT,
                                 target=openalex_api.DB_TARGET, mailto=mailto)
    else:
        openalex_api.fetch_jsonl(folder / "cs_works.jsonl",
                                 filter_value=openalex_api.CS_FILTER,
                                 select=openalex_api.CS_SELECT,
                                 target=openalex_api.CS_TARGET, mailto=mailto)


SOURCES = {
    "movielens_1m": "movielens",
    "pokec_hobby": "pokec",
    "imdb_cast": "imdb",
    "imdb_cast_full": "imdb",
    "acs_pums_household": "acs_pums",
    "openalex_db": "openalex",
    "openalex_cs_size3plus": "openalex",
}
ALL = ["movielens_1m", "pokec_hobby", "imdb_cast", "acs_pums_household"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=sorted(SOURCES) + ["all"])
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--states", nargs="+", default=["ma"], help="ACS PUMS states (paper: ma)")
    ap.add_argument("--mailto", default=None,
                    help="e-mail for the OpenAlex polite pool (optional; no default)")
    a = ap.parse_args()
    todo = ALL if a.dataset == "all" else [a.dataset]
    done: set[str] = set()
    for name in todo:
        src = SOURCES[name]
        if src in done and src != "openalex":
            continue
        done.add(src)
        print(f"[{name}] licence: {LICENCES[src]}")
        if src == "movielens":
            get_movielens(a.raw_dir)
        elif src == "pokec":
            get_pokec(a.raw_dir)
        elif src == "imdb":
            get_imdb(a.raw_dir)
        elif src == "acs_pums":
            get_acs(a.raw_dir, [s.lower() for s in a.states])
        else:
            get_openalex(a.raw_dir, name, a.mailto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
