"""Inventario de archivos de audio del corpus Doppler."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SIRENNET_CLASS_MAP = {
    "ambulance": "ambulance",
    "police": "police",
    "firetruck": "firetruck",
    "fire_truck": "firetruck",
    "fire": "firetruck",
    "traffic": "traffic",
}

LSSIREN_POSITIVE_HINTS = ("emergency", "siren", "ambulance")
LSSIREN_NEGATIVE_HINTS = ("road", "noise", "traffic")
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_SUFFIXES


def infer_sirennet_label(path: Path) -> str | None:
    parts = [p.lower() for p in path.parts]
    stem = path.stem.lower()
    for key, label in SIRENNET_CLASS_MAP.items():
        if key in parts or stem.startswith(key) or f"_{key}_" in f"_{stem}_":
            return label
    return None


def infer_lssiren_label(path: Path) -> str | None:
    blob = " ".join(p.lower() for p in path.parts)
    if any(h in blob for h in LSSIREN_POSITIVE_HINTS) and "road" not in blob:
        return "siren"
    if any(h in blob for h in LSSIREN_NEGATIVE_HINTS):
        return "road_noise"
    return None


def scan_sirennet(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.rglob("*"):
        if not path.is_file() or not is_audio(path):
            continue
        rows.append(
            {
                "corpus": "sirennet",
                "path": str(path),
                "relpath": str(path.relative_to(root)),
                "label": infer_sirennet_label(path) or "unknown",
                "task": "multiclass",
            }
        )
    return pd.DataFrame(rows)


LSSIREN_CSV_COLUMNS = [
    "filename",
    "chroma_stft",
    "rmse",
    "spectral_centroid",
    "spectral_bandwidth",
    "rolloff",
    "zero_crossing_rate",
    *[f"mfcc{i}" for i in range(1, 21)],
    "label",
]


def read_lssiren_features(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if "filename" in raw.columns:
        return raw
    return pd.read_csv(path, header=None, names=LSSIREN_CSV_COLUMNS)


def scan_lssiren(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.rglob("*"):
        if not path.is_file() or not is_audio(path):
            continue
        label = infer_lssiren_label(path) or "unknown"
        rows.append(
            {
                "corpus": "lssiren",
                "path": str(path),
                "relpath": str(path.relative_to(root)),
                "label": label,
                "task": "binary",
            }
        )
    if rows:
        return pd.DataFrame(rows)

    for csv_path in root.glob("*.csv"):
        feat = read_lssiren_features(csv_path)
        label_col = "label" if "label" in feat.columns else feat.columns[-1]
        name_col = "filename" if "filename" in feat.columns else feat.columns[0]
        for _, row in feat.iterrows():
            raw_label = str(row[label_col]).strip().lower()
            label = "siren" if raw_label in {"ambulance", "siren", "emergency"} else "road_noise"
            rows.append(
                {
                    "corpus": "lssiren",
                    "path": str(csv_path.parent / str(row[name_col])),
                    "relpath": str(row[name_col]),
                    "label": label,
                    "task": "binary",
                    "source": "feature_csv",
                }
            )
    return pd.DataFrame(rows)


def scan_urbansound8k(root: Path) -> pd.DataFrame:
    csv_candidates = list(root.rglob("UrbanSound8K.csv"))
    if csv_candidates:
        meta = pd.read_csv(csv_candidates[0])
        if "class" not in meta.columns and "class_name" in meta.columns:
            meta = meta.rename(columns={"class_name": "class"})
        audio_root = csv_candidates[0].parent.parent / "audio"
        if not audio_root.exists():
            audio_root = root / "audio"
            if not audio_root.exists():
                audio_root = root
        rows = []
        for _, row in meta.iterrows():
            fold = int(row["fold"])
            fname = row["slice_file_name"]
            path = audio_root / f"fold{fold}" / fname
            rows.append(
                {
                    "corpus": "urbansound8k",
                    "path": str(path),
                    "relpath": f"fold{fold}/{fname}",
                    "label": row["class"],
                    "task": "urban_scene",
                    "fold": fold,
                    "fsID": row.get("fsID"),
                    "classID": row.get("classID"),
                    "salience": row.get("salience"),
                }
            )
        return pd.DataFrame(rows)

    rows = []
    for path in root.rglob("*"):
        if not path.is_file() or not is_audio(path):
            continue
        rows.append(
            {
                "corpus": "urbansound8k",
                "path": str(path),
                "relpath": str(path.relative_to(root)),
                "label": path.parent.name,
                "task": "urban_scene",
            }
        )
    return pd.DataFrame(rows)
