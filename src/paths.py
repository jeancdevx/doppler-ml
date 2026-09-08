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
    "audioset_ev": (
        "audioset_ev",
        "audioset-ev",
        "audioset_ev_v2",
        "audioset-ev-v2",
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


def _input_dirs() -> list[Path]:
    root = data_root()
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir()]


def _dir_names(path: Path) -> set[str]:
    try:
        return {p.name.lower() for p in path.iterdir()}
    except OSError:
        return set()


def _looks_like_sirennet(path: Path) -> bool:
    names = _dir_names(path)
    return {"ambulance", "police"}.issubset(names) or {"ambulance", "firetruck", "traffic"}.issubset(names)


def _looks_like_lssiren(path: Path) -> bool:
    names = _dir_names(path)
    if "ambulance_final.csv" in names or "road_final.csv" in names:
        return True
    return any("emergency" in n and "siren" in n for n in names) or "road noises" in names


def _looks_like_urbansound8k(path: Path) -> bool:
    if (path / "UrbanSound8K.csv").exists() or (path / "metadata" / "UrbanSound8K.csv").exists():
        return True
    try:
        return any(p.name == "UrbanSound8K.csv" for p in path.rglob("UrbanSound8K.csv"))
    except OSError:
        return False


def _looks_like_audioset_ev(path: Path) -> bool:
    names = _dir_names(path)
    if "positive_files" in names or "ev_positives.csv" in names:
        return True
    return (path / "EV_Positives.csv").exists() or (path / "Positive_files").is_dir()


SIGNATURES = {
    "sirennet": _looks_like_sirennet,
    "lssiren": _looks_like_lssiren,
    "urbansound8k": _looks_like_urbansound8k,
    "audioset_ev": _looks_like_audioset_ev,
}


def resolve_corpus(name: str) -> Path | None:
    """Devuelve el directorio del corpus si está montado."""
    slugs = KAGGLE_SLUGS[name]
    slug_candidates: list[Path] = []
    if is_kaggle():
        for slug in slugs:
            slug_candidates.append(KAGGLE_INPUT / slug)
        datasets = KAGGLE_INPUT / "datasets"
        if datasets.exists():
            for user_dir in datasets.iterdir():
                if not user_dir.is_dir():
                    continue
                for slug in slugs:
                    slug_candidates.append(user_dir / slug)
                    slug_candidates.append(user_dir / slug / slug)
        for ds in _input_dirs():
            slug_candidates.append(ds / name)
            for slug in slugs:
                slug_candidates.append(ds / slug)
    else:
        slug_candidates.append(LOCAL_RAW / name)
        for slug in slugs:
            slug_candidates.append(LOCAL_RAW / slug)

    existing = [path for path in slug_candidates if path.exists()]
    if existing:
        return max(existing, key=lambda path: len(path.parts))

    for ds in _input_dirs():
        if SIGNATURES[name](ds):
            return ds
        try:
            children = list(ds.iterdir())
        except OSError:
            children = []
        for child in children:
            if child.is_dir() and SIGNATURES[name](child):
                return child
    return None


@dataclass(frozen=True)
class CorpusPaths:
    sirennet: Path | None
    lssiren: Path | None
    urbansound8k: Path | None
    audioset_ev: Path | None

    def available(self) -> dict[str, Path]:
        found = {
            "sirennet": self.sirennet,
            "lssiren": self.lssiren,
            "urbansound8k": self.urbansound8k,
            "audioset_ev": self.audioset_ev,
        }
        return {key: path for key, path in found.items() if path is not None}


def corpus_paths() -> CorpusPaths:
    return CorpusPaths(
        sirennet=resolve_corpus("sirennet"),
        lssiren=resolve_corpus("lssiren"),
        urbansound8k=resolve_corpus("urbansound8k"),
        audioset_ev=resolve_corpus("audioset_ev"),
    )
