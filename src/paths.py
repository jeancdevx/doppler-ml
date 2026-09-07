"""Resolución de rutas local vs Kaggle para el corpus Doppler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RAW = REPO_ROOT / "data" / "raw"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
TABLES_DIR = REPO_ROOT / "reports" / "tables"

# Nombres esperados al subir/adjuntar datasets en Kaggle.
KAGGLE_SLUGS = {
    "sirennet": (
        "sirennet",
        "sirennet-emergency-vehicle-siren-classification",
        "j4ydzzv4kb",
    ),
    "lssiren": (
        "lssiren",
        "large-scale-audio-dataset-for-emergency-vehicle-sirens",
        "emergency-vehicle-sirens-and-road-noises",
    ),
    "urbansound8k": (
        "urbansound8k",
        "urban-sound-8k",
        "urbansound8k-dataset",
    ),
}


def is_kaggle() -> bool:
    return KAGGLE_INPUT.exists()


def data_root() -> Path:
    return KAGGLE_INPUT if is_kaggle() else LOCAL_RAW


def figures_dir() -> Path:
    if is_kaggle():
        path = Path("/kaggle/working/reports/figures")
    else:
        path = FIGURES_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def tables_dir() -> Path:
    if is_kaggle():
        path = Path("/kaggle/working/reports/tables")
    else:
        path = TABLES_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_corpus(name: str) -> Path | None:
    """Devuelve el directorio del corpus si está montado."""
    root = data_root()
    slugs = KAGGLE_SLUGS[name]
    candidates: list[Path] = []
    if is_kaggle():
        for slug in slugs:
            candidates.append(root / slug)
    else:
        candidates.append(LOCAL_RAW / name)
        for slug in slugs:
            candidates.append(LOCAL_RAW / slug)
    return _first_existing(candidates)


@dataclass(frozen=True)
class CorpusPaths:
    sirennet: Path | None
    lssiren: Path | None
    urbansound8k: Path | None

    def available(self) -> dict[str, Path]:
        found = {
            "sirennet": self.sirennet,
            "lssiren": self.lssiren,
            "urbansound8k": self.urbansound8k,
        }
        return {key: path for key, path in found.items() if path is not None}


def corpus_paths() -> CorpusPaths:
    return CorpusPaths(
        sirennet=resolve_corpus("sirennet"),
        lssiren=resolve_corpus("lssiren"),
        urbansound8k=resolve_corpus("urbansound8k"),
    )
