"""Download the official NeuralOperator 2D Navier--Stokes dataset.

Dataset DOI: 10.5281/zenodo.12825163
Official loader: neuraloperator/neuraloperator, neuralop/data/datasets/navier_stokes.py

The archive is stored outside Git by default.  This script verifies the MD5
published by Zenodo before extraction and rejects archive paths that would
escape the requested data directory.
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

FILES = {
    128: {
        "name": "nsforcing_128.tgz",
        "md5": "70a389207ac93935d5ff3d4289d43581",
        "size_gb": 1.5,
    },
    1024: {
        "name": "nsforcing_1024.tgz",
        "md5": "8b5c239a215e5e9ff0863fed2e8adcdf",
        "size_gb": 15.4,
    },
}
ZENODO_RECORD = "12825163"


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "dcurc-ns2d/0.1"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as stream:
        total = int(response.headers.get("Content-Length", "0"))
        received = 0
        while chunk := response.read(8 * 1024 * 1024):
            stream.write(chunk)
            received += len(chunk)
            if total:
                print(f"\rDownloaded {received / total:6.1%}", end="", flush=True)
    print()
    partial.replace(destination)


def safe_extract(archive: Path, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (root / member.name).resolve()
            if resolved_root not in target.parents and target != resolved_root:
                raise ValueError(f"Unsafe archive member: {member.name}")
        if hasattr(tarfile, "data_filter"):
            bundle.extractall(root, filter="data")
        else:  # Python 3.10--3.11 compatibility after explicit path validation.
            bundle.extractall(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/ns2d"))
    parser.add_argument("--resolution", type=int, choices=FILES, default=128)
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    metadata = FILES[args.resolution]
    archive = args.root / str(metadata["name"])
    url = (
        f"https://zenodo.org/records/{ZENODO_RECORD}/files/"
        f"{metadata['name']}?download=1"
    )
    print(
        f"Official NS2D {args.resolution} archive: approximately "
        f"{metadata['size_gb']} GB"
    )
    if not archive.exists():
        download(url, archive)
    observed = md5sum(archive)
    if observed != metadata["md5"]:
        raise RuntimeError(
            f"MD5 mismatch for {archive}: expected {metadata['md5']}, got {observed}"
        )
    print(f"MD5 verified: {observed}")
    safe_extract(archive, args.root)
    if not args.keep_archive:
        archive.unlink()
    print(f"Dataset ready at {args.root.resolve()}")


if __name__ == "__main__":
    main()
