"""Download the upstream code that the baselines import, at the recorded commits.

    python baselines/fetch_third_party.py            # all three
    python baselines/fetch_third_party.py ahcka cahc # a subset

Each repository is downloaded as a GitHub archive of the pinned commit and
unpacked into baselines/third_party/<name>/. The files the run scripts import
are checked against SHA-256 digests taken from the copies that produced the
audited partitions. These repositories carry no license file, so their code
is fetched, not redistributed here. GRAC's Laplacian (MIT) is vendored in
third_party/grac_cikm/ and needs no download.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

THIRD_PARTY = Path(__file__).resolve().parent / "third_party"

REPOS = {
    "ahcka": {
        "url": "https://github.com/CyanideCentral/AHCKA",
        "commit": "e65f459701ae61a5794a36eab2bbf02b1b4fe2e4",
        "sha256": {
            "cluster.py": "000538cc5a52b81e71ef39a9f3c0d9ec54c99d9b4aefda7e44e996fc2b2aa1d7",
            "config.py": "bc2de16023814f26192cfa3507170f049b73ae954d6792623c7ab20757689f35",
            "spectral.py": "dacb35fecca4bf79a044e5d7ddb172b16cf551b9b3d4908eb18a681db4f8d1a3",
            "munkres.py": "c717264184d681b91de80cc0a2be9a127a91279e36b25fa4777a4e4b4e19e226",
        },
    },
    "cahc": {
        "url": "https://github.com/nilics/CAHC",
        "commit": "e6975ea33b0734255137e7592f83f7fcd325405d",
        "sha256": {
            "model/Augmentor.py": "03c16dfc0a7d9b90fdb71079db3450bacd9ff525dab2d9df84f2a30c78ec58a3",
            "model/ClusteringLayer.py": "63648eef428791e12aa935fa789194a5ace7a179dd3bea52b60988eb09603e24",
            "model/EnhancedHGNN.py": "9be05cea0f557cae30d237f12cb4624dfdd620e31da9d33dbb4a93c9d5319686",
            "model/EnhancedHGNNConv.py": "27b78109b6a8c14e0378994722355d16f089576fba00316980868c1ca26b244b",
            "model/Evaluator.py": "977b6f4bf04cf990ce502295b33a5e01415f5702a0fd8d9998530406bd541330",
            "model/Loss.py": "667fc5f331ad344065a352bee38026f694bc09e475754f188c368161f34fe2af",
            "model/data_loader.py": "25a783eb08423356a6cd59519d4c07d003a74396f922f33f279bddfa07022bb7",
            "model/dataset.py": "7215bee270d9d1313e178fc0539884e084f057d287ebd42b1f70ad44b084293f",
            "model/hgnn.py": "bae15e4d846ab238856cbb577df4b8a0bb15dc4b63887dae26b8ed45aeb77510",
            "model/train.py": "d8f5bcefb89475b890df11f4b278afe8245ed083576ee91ddfff1ea1927d22e6",
        },
    },
    "shyre": {
        "url": "https://github.com/Abel0828/supervised_hypergraph_reconstruction",
        "commit": "d49c77ed2ef5d41d5c1707211fcf17e2a2283830",
        "sha256": {
            "utils.py": "3d54508423be0f949ddd8a55efb3882e801ff1785f934987d93b2e72b4873e4b",
            "cliques.py": "76f35b5303efe7cd5a2ee25bcb3b17a44517e6168bd408e0b1d9efba98368c5d",
            "train.py": "51fa917fa672107967ab2a48c76a931e921e7cd6eb2550a1699d67effae6236d",
            "model.py": "b3d676775ca8895eaef0834aa09b1c305a205a9d1c85b04ff06f24a7578c1564",
            "motif.py": "129ffb9059e37ef755c1f5238cf0d9f279dc2c5416a11393042c6f816069cd6b",
            "baselines/ecc.py": "1a4bea72f34b62e5a95e5c38386e71786adcdf1f591d0917c0f373cb8bf0c92d",
            "baselines/community.py": "47df20bffdd261b475cd4a2b2c70aae0fa3115d044a714e5e6a64da0e79cfa35",
            "baselines/beyesian.py": "e223a75a17ccb65eade1ad2657b120365086adf3d3329492af737b94030486e5",
        },
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(name: str) -> list[str]:
    target = THIRD_PARTY / name
    bad = []
    for rel, digest in REPOS[name]["sha256"].items():
        f = target / rel
        if not f.exists():
            bad.append(f"{rel}: missing")
        elif sha256(f) != digest:
            bad.append(f"{rel}: SHA-256 differs")
    return bad


def fetch(name: str, force: bool) -> None:
    spec = REPOS[name]
    target = THIRD_PARTY / name
    if target.exists() and not force:
        bad = verify(name)
        if not bad:
            print(f"[{name}] already present at commit {spec['commit'][:12]}")
            return
        raise SystemExit(f"[{name}] {target} exists but does not match: {bad}; rerun with --force")
    owner_repo = spec["url"].split("github.com/")[1]
    url = f"https://codeload.github.com/{owner_repo}/zip/{spec['commit']}"
    print(f"[{name}] downloading {spec['url']} at {spec['commit']}", flush=True)
    with urllib.request.urlopen(url, timeout=600) as resp:
        blob = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(blob))
    if zf.comment.decode("ascii", "ignore") != spec["commit"]:
        raise SystemExit(f"[{name}] archive is not commit {spec['commit']}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for info in zf.infolist():
        parts = Path(info.filename).parts
        rel = Path(*parts[1:]) if len(parts) > 1 else None
        if rel is None or info.is_dir():
            continue
        if rel.is_absolute() or ".." in rel.parts:
            raise SystemExit(f"[{name}] unsafe path in archive: {info.filename}")
        out = target / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(zf.read(info))
    bad = verify(name)
    if bad:
        raise SystemExit(f"[{name}] checksum check failed: {bad}")
    print(f"[{name}] ok -> {target}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help=f"any of {', '.join(REPOS)} (default: all)")
    p.add_argument("--force", action="store_true", help="replace an existing folder")
    a = p.parse_args()
    unknown = [x for x in a.names if x not in REPOS]
    if unknown:
        p.error(f"unknown names {unknown}; choose from {list(REPOS)}")
    for name in a.names or list(REPOS):
        fetch(name, a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
