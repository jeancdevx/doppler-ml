#!/usr/bin/env python3
"""Descarga los corpora de trabajo del proyecto Doppler."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
USER_AGENT = "doppler-ml/0.1 (academic research; +https://github.com)"

MENDELEY_DATASET_API = "https://data.mendeley.com/public-api/datasets/j4ydzzv4kb"
FIGSHARE_API = "https://api.figshare.com/v2/articles/19291472"
URBANSOUND_URL = "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1"
URBANSOUND_META = "https://raw.githubusercontent.com/Raghad-th/Classifying-Urban-Sounds-Using-Deep-Learning/main/metadata/UrbanSound8K.csv"


def _request(url: str, timeout: int = 120):
    req = Request(url, headers={"User-Agent": USER_AGENT})
    return urlopen(req, timeout=timeout)


def download_file(url: str, dest: Path, timeout: int = 600) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f" -> {dest}")
    with _request(url, timeout=timeout) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh)
    print(f"Saved {dest.stat().st_size / 1e6:.1f} MB")
    return dest


def extract_archive(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(target)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
    else:
        raise ValueError(f"Unsupported archive: {archive}")


def flatten_single_root(target: Path) -> None:
    children = [p for p in target.iterdir() if p.name not in {".", ".."}]
    if len(children) == 1 and children[0].is_dir():
        inner = children[0]
        for item in inner.iterdir():
            dest = target / item.name
            if dest.exists():
                continue
            shutil.move(str(item), dest)
        inner.rmdir()


def download_sirennet() -> Path:
    dest_dir = RAW / "sirennet"
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = RAW / "_archives" / "sirennet.zip"
    with _request(MENDELEY_DATASET_API, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    files = payload.get("files", [])
    if not files:
        raise RuntimeError("Mendeley API returned no files for sireNNet")
    url = files[0]["content_details"]["download_url"]
    print(f"sireNNet zip: {files[0]['filename']} ({files[0]['size'] / 1e6:.1f} MB)")
    download_file(url, archive, timeout=1800)
    extract_archive(archive, dest_dir)
    flatten_single_root(dest_dir)
    return dest_dir


def download_lssiren(*, csv_only: bool = False) -> Path:
    dest_dir = RAW / "lssiren"
    dest_dir.mkdir(parents=True, exist_ok=True)
    with _request(FIGSHARE_API, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    files = payload.get("files", [])
    print(f"Figshare API returned {len(files)} file(s)")
    for item in files:
        name = item.get("name", "")
        print(f"  - {name} ({item.get('size', 0) / 1e6:.1f} MB)")
        if csv_only and not name.lower().endswith(".csv"):
            continue
        out = dest_dir / name
        download_file(item["download_url"], out, timeout=1800)
        if name.endswith(".zip"):
            extract_archive(out, dest_dir)
    flatten_single_root(dest_dir)
    return dest_dir


def download_urbansound8k(*, metadata_only: bool = False) -> Path:
    dest_dir = RAW / "urbansound8k"
    dest_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = dest_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    download_file(URBANSOUND_META, meta_dir / "UrbanSound8K.csv", timeout=60)
    if metadata_only:
        print("UrbanSound8K: metadata only (skip ~5.6 GB audio archive)")
        return dest_dir
    archive = RAW / "_archives" / "UrbanSound8K.tar.gz"
    download_file(URBANSOUND_URL, archive, timeout=1800)
    extract_archive(archive, dest_dir)
    flatten_single_root(dest_dir)
    return dest_dir


CORPUS_FNS = {
    "sirennet": download_sirennet,
    "lssiren": download_lssiren,
    "urbansound8k": download_urbansound8k,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Doppler working corpora")
    parser.add_argument(
        "--corpus",
        choices=[*CORPUS_FNS, "all"],
        default="all",
    )
    parser.add_argument(
        "--lssiren-csv-only",
        action="store_true",
        help="Download LSSiren feature CSVs without the ~2.3 GB WAV archives.",
    )
    parser.add_argument(
        "--urbansound-metadata-only",
        action="store_true",
        help="Download UrbanSound8K.csv without the 5.6 GB audio archive.",
    )
    args = parser.parse_args()
    names = list(CORPUS_FNS) if args.corpus == "all" else [args.corpus]
    for name in names:
        print(f"\n=== {name} ===")
        if name == "urbansound8k":
            path = download_urbansound8k(metadata_only=args.urbansound_metadata_only)
        elif name == "lssiren":
            path = download_lssiren(csv_only=args.lssiren_csv_only)
        else:
            path = CORPUS_FNS[name]()
        wavs = list(path.rglob("*.wav"))
        print(f"{name}: {path} ({len(wavs)} wav files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
