"""Bounded OpenAlex /works download into a JSONL cache (one work per line).

Used by ``data/download.py --dataset openalex_db|openalex_cs_size3plus``.

* ``mailto`` (OpenAlex "polite pool" e-mail) is passed on the command line;
  there is no default.
* OpenAlex requires a free API key for cursor paging past the first page
  (as of October 2026). It is read from the environment variable
  ``OPENALEX_API_KEY`` and is never printed or written to disk.

The cache is resumable: works already in the file are skipped and the fetch
stops once ``target`` unique works are cached or the query is exhausted.
OpenAlex changes over time, so a new fetch does not reproduce the snapshot
used in the paper (fetched in May 2026); the processed OpenAlex datasets
therefore ship with the repository.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.openalex.org/works"
USER_AGENT = "release-determinacy-ahc-data/1.0"

# OA-DB: works whose primary location is one of these six OpenAlex source
# records, used in the paper for the SIGMOD, VLDB, ICDE and PODS venues.
DB_SOURCE_IDS = ["S4306419648", "S4210226185", "S4306420993", "S4306418077",
                 "S4363608845", "S4210210321"]
DB_FILTER = "primary_location.source.id:" + "|".join(DB_SOURCE_IDS)
DB_SELECT = "id,publication_year,authorships,concepts,topics,primary_topic,primary_location"
DB_TARGET = 1_000_000      # all works; the paper cache held the 6,035 returned in May 2026

# OA-CS-3+: works tagged with the Computer Science concept, 2018-2024.
CS_CONCEPT = "C41008148"
CS_FILTER = f"concepts.id:{CS_CONCEPT},publication_year:2018-2024"
CS_SELECT = "id,publication_year,authorships,concepts,topics,primary_topic"
CS_TARGET = 30000          # the paper cache held the first 30,000 works returned


def fetch_jsonl(cache_file: Path, *, filter_value: str, select: str, target: int,
                mailto: str | None = None, per_page: int = 100,
                sleep_seconds: float = 0.2) -> int:
    cache_file = Path(cache_file)
    if target < 1:
        raise ValueError("target must be positive")
    seen: set[str] = set()
    if cache_file.exists():
        with cache_file.open(encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    seen.add(json.loads(line)["id"])
    if len(seen) >= target:
        print(f"{cache_file.name}: already holds {len(seen)} works")
        return len(seen)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT}
    api_key = os.environ.get("OPENALEX_API_KEY")
    cursor = "*"
    with cache_file.open("a", encoding="utf-8") as out:
        while len(seen) < target:
            params = {"filter": filter_value, "select": select, "per_page": per_page,
                      "cursor": cursor}
            if mailto:
                params["mailto"] = mailto
            if api_key:
                params["api_key"] = api_key
            request = urllib.request.Request(API + "?" + urllib.parse.urlencode(params),
                                             headers=headers)
            payload = None
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(request, timeout=60) as response:
                        payload = json.load(response)
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code == 401:
                        raise SystemExit(
                            "OpenAlex HTTP 401: paging past the first page needs a free API "
                            "key (https://openalex.org/settings/api). Set it in the "
                            "environment variable OPENALEX_API_KEY and rerun; the cache "
                            "resumes where it stopped.") from None
                    if exc.code not in (429, 500, 502, 503, 504) or attempt == 4:
                        raise RuntimeError(f"OpenAlex HTTP {exc.code}") from None
                    time.sleep(min(2 ** attempt, 16))
                except (urllib.error.URLError, TimeoutError):
                    if attempt == 4:
                        raise RuntimeError("OpenAlex request failed after 5 attempts") from None
                    time.sleep(min(2 ** attempt, 16))
            rows = payload.get("results", []) if payload else []
            if not rows:
                break
            for row in rows:
                if row["id"] in seen:
                    continue
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                seen.add(row["id"])
                if len(seen) >= target:
                    break
            out.flush()
            next_cursor = payload.get("meta", {}).get("next_cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
            time.sleep(sleep_seconds)
    print(f"{cache_file.name}: {len(seen)} unique works cached")
    return len(seen)
