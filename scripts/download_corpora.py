#!/usr/bin/env python3
"""Descarga los corpora de trabajo del proyecto Doppler."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
USER_AGENT = "doppler-ml/0.1 (academic research; +https://github.com)"

MENDELEY_DATASET_API = "https://data.mendeley.com/public-api/datasets/j4ydzzv4kb"
FIGSHARE_API = "https://api.figshare.com/v2/articles/19291472"
URBANSOUND_URL = "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1"
URBANSOUND_META = "https://raw.githubusercontent.com/Raghad-th/Classifying-Urban-Sounds-Using-Deep-Learning/main/metadata/UrbanSound8K.csv"
AUDIOSET_EV_API = "https://zenodo.org/api/records/18668076"


def _request(url: str, timeout: int = 120):
    req = Request(url, headers={"User-Agent": USER_AGENT})
    return urlopen(req, timeout=timeout)


def download_file(url: str, dest: Path, timeout: int = 600) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    print(f"Downloading {url}")
    print(f" -> {dest}")
    headers = {"User-Agent": USER_AGENT}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"Resuming from {existing / 1e9:.2f} GB")
    req = Request(url, headers=headers)
    try:
        resp_cm = urlopen(req, timeout=timeout)
    except HTTPError as exc:
        if exc.code == 416 and dest.exists():
            print("Already complete (HTTP 416)")
            return dest
        raise
    with resp_cm as resp:
        status = getattr(resp, "status", 200)
        mode = "ab"
        if existing and status == 200:
            print("Server ignored Range; rewriting file")
            mode = "wb"
        elif not existing:
            mode = "wb"
        with dest.open(mode) as fh:
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


def _free_gb() -> float:
    usage = shutil.disk_usage(RAW)
    return usage.free / 1e9


def _zip_uncompressed_gb(archive: Path) -> float:
    total = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            total += info.file_size
    return total / 1e9


class _HttpRangeFile:
    """Archivo seekable sobre HTTP Range (para leer un zip remoto sin bajarlo entero)."""

    def __init__(self, url: str, block: int = 4 * 1024 * 1024):
        self.url = url
        self.pos = 0
        self.block = block
        self._buf = b""
        self._buf_start = 0
        req = Request(url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
        with urlopen(req, timeout=120) as resp:
            cr = resp.headers.get("Content-Range", "")
            if "/" in cr:
                self.size = int(cr.rsplit("/", 1)[1])
            else:
                self.size = int(resp.headers.get("Content-Length") or 0)
            self.url = resp.geturl() or url
        if self.size <= 0:
            raise RuntimeError(f"Could not determine size of {url}")

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        else:
            raise ValueError(whence)
        return self.pos

    def tell(self) -> int:
        return self.pos

    def _fetch(self, start: int, n: int) -> bytes:
        if n <= 0 or start >= self.size:
            return b""
        end = min(start + n, self.size) - 1
        last_exc: Exception | None = None
        for attempt in range(5):
            req = Request(
                self.url,
                headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"},
            )
            try:
                with urlopen(req, timeout=300) as resp:
                    return resp.read()
            except Exception as exc:
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        raise last_exc  # type: ignore[misc]

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        buf_end = self._buf_start + len(self._buf)
        if self.pos < self._buf_start or self.pos >= buf_end:
            grab = max(n, self.block)
            self._buf = self._fetch(self.pos, grab)
            self._buf_start = self.pos
            buf_end = self._buf_start + len(self._buf)
        offset = self.pos - self._buf_start
        chunk = self._buf[offset : offset + n]
        if len(chunk) < n:
            extra = self._fetch(self.pos + len(chunk), n - len(chunk))
            chunk += extra
            self._buf = chunk
            self._buf_start = self.pos
        self.pos += len(chunk)
        return chunk


def _audioset_zip_url() -> tuple[str, str, int]:
    with _request(AUDIOSET_EV_API, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    entries = payload.get("files", [])
    if not entries:
        raise RuntimeError("Zenodo API returned no files for AudioSet-EV v2")
    item = max(entries, key=lambda row: int(row.get("size") or 0))
    url = item["links"].get("content") or item["links"]["self"]
    return url, item["key"], int(item["size"])


def extract_audioset_preview(*, max_wav: int = 240) -> Path:
    """CSV completos + muestra de WAV vía Range, sin esperar el zip de 16 GB."""
    dest_dir = RAW / "audioset_ev"
    dest_dir.mkdir(parents=True, exist_ok=True)
    url, key, size = _audioset_zip_url()
    print(f"Preview extract from {key} ({size / 1e9:.2f} GB) via HTTP Range")
    remote = _HttpRangeFile(url)
    with zipfile.ZipFile(remote) as zf:
        names = zf.namelist()
        csv_names = [n for n in names if n.lower().endswith(".csv")]
        wav_names = [n for n in names if n.lower().endswith(".wav")]
        print(f"Remote members: {len(names)} ({len(wav_names)} wav, {len(csv_names)} csv)")
        for name in csv_names:
            out = dest_dir / Path(name).name
            nested = dest_dir / name
            if out.exists() or nested.exists():
                print(f"  csv skip existing {name}")
                continue
            zf.extract(name, dest_dir)
            print(f"  csv {name}")

        buckets: dict[str, list[str]] = defaultdict(list)
        for name in wav_names:
            key = name.replace("\\", "/")
            if "Positive_files" in key and "/unbalanced/" in key:
                buckets["pos_unb"].append(name)
            elif "Positive_files" in key and "/eval/" in key:
                buckets["pos_eval"].append(name)
            elif "Positive_files" in key and "/balanced_train/" in key:
                buckets["pos_bal"].append(name)
            elif "Negative_files" in key and "/eval/" in key:
                buckets["neg_eval"].append(name)
            elif "Negative_files" in key and "/balanced_train/" in key:
                buckets["neg_bal"].append(name)
            else:
                buckets["other"].append(name)
        quotas = {
            "pos_unb": 50,
            "pos_eval": 40,
            "pos_bal": 40,
            "neg_bal": 50,
            "neg_eval": 40,
            "other": 20,
        }
        sample: list[str] = []
        for key, quota in quotas.items():
            sample.extend(buckets[key][:quota])
        sample = sample[:max_wav]
        for i, name in enumerate(sample, start=1):
            wav_dest = dest_dir / name
            if wav_dest.exists() or (dest_dir / Path(name).name).exists():
                if i % 20 == 0 or i == len(sample):
                    print(f"  wav skip {i}/{len(sample)}")
                continue
            zf.extract(name, dest_dir)
            if i % 20 == 0 or i == len(sample):
                print(f"  wav {i}/{len(sample)}")
    flatten_single_root(dest_dir)
    return dest_dir


def download_audioset_ev(*, metadata_only: bool = False) -> Path:
    dest_dir = RAW / "audioset_ev"
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = dest_dir / "Positive_files"
    if marker.exists() and not metadata_only:
        n_wav = len(list(dest_dir.rglob("*.wav")))
        print(f"AudioSet-EV already extracted ({n_wav} wav). Skip download.")
        return dest_dir

    with _request(AUDIOSET_EV_API, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    entries = payload.get("files", [])
    if not entries:
        raise RuntimeError("Zenodo API returned no files for AudioSet-EV v2")
    item = max(entries, key=lambda row: int(row.get("size") or 0))
    url = item["links"].get("content") or item["links"]["self"]
    size_gb = int(item["size"]) / 1e9
    print(f"AudioSet-EV zip: {item['key']} ({size_gb:.2f} GB)")
    print(f"Free disk: {_free_gb():.1f} GB")

    leftover = RAW / "sirennet.zip"
    if leftover.exists():
        print(f"Removing leftover {leftover} ({leftover.stat().st_size / 1e6:.0f} MB)")
        leftover.unlink()

    archive = RAW / "_archives" / item["key"]
    download_file(url, archive, timeout=7200)

    uncompressed_gb = _zip_uncompressed_gb(archive)
    print(f"Uncompressed listing: {uncompressed_gb:.2f} GB; free {_free_gb():.1f} GB")

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        csv_names = [n for n in names if n.lower().endswith(".csv")]
        wav_names = [n for n in names if n.lower().endswith(".wav")]
        print(f"Archive members: {len(names)} ({len(wav_names)} wav, {len(csv_names)} csv)")
        for name in csv_names:
            zf.extract(name, dest_dir)
        flatten_single_root(dest_dir)

        if metadata_only:
            print("AudioSet-EV: metadata only (skip WAV extract)")
            return dest_dir

        need_gb = uncompressed_gb + 1.0
        if _free_gb() < need_gb:
            print(
                f"Not enough free disk to extract all WAV ({need_gb:.1f} GB needed). "
                "Extracting a stratified sample for EDA; keep the zip."
            )
            sample = [n for n in wav_names if "/balanced_train/" in n or "\\balanced_train\\" in n]
            rest = [n for n in wav_names if n not in sample]
            sample.extend(rest[:400])
            for i, name in enumerate(sample, start=1):
                zf.extract(name, dest_dir)
                if i % 100 == 0:
                    print(f"  extracted {i}/{len(sample)}")
            flatten_single_root(dest_dir)
            return dest_dir

        print("Extracting all AudioSet-EV WAV (this takes a while)...")
        zf.extractall(dest_dir)
    flatten_single_root(dest_dir)
    print(f"Removing archive {archive}")
    archive.unlink(missing_ok=True)
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
    "audioset_ev": download_audioset_ev,
}
WORKING_CORPORA = ("sirennet", "lssiren", "urbansound8k")


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
    parser.add_argument(
        "--audioset-metadata-only",
        action="store_true",
        help="Extract AudioSet-EV CSVs without unpacking ~16 GB of WAV.",
    )
    parser.add_argument(
        "--audioset-preview",
        action="store_true",
        help="Extract AudioSet-EV CSVs plus a small WAV sample via HTTP Range (no 16 GB zip).",
    )
    args = parser.parse_args()
    names = list(WORKING_CORPORA) if args.corpus == "all" else [args.corpus]
    for name in names:
        print(f"\n=== {name} ===")
        if name == "urbansound8k":
            path = download_urbansound8k(metadata_only=args.urbansound_metadata_only)
        elif name == "lssiren":
            path = download_lssiren(csv_only=args.lssiren_csv_only)
        elif name == "audioset_ev":
            if args.audioset_preview:
                path = extract_audioset_preview()
            else:
                path = download_audioset_ev(metadata_only=args.audioset_metadata_only)
        else:
            path = CORPUS_FNS[name]()
        wavs = list(path.rglob("*.wav"))
        print(f"{name}: {path} ({len(wavs)} wav files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
