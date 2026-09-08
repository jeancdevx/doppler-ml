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


AUDIOSET_EV_MIDS = {
    "/m/04qvtq": "police",
    "/m/012n7d": "ambulance",
    "/m/012ndj": "firetruck",
}
AUDIOSET_EV_GENERIC_MIDS = {
    "/m/03j1ly",  # Emergency vehicle
    "/m/03kmc9",  # Siren
}


def parse_audioset_mids(raw) -> list[str]:
    if raw is None:
        return []
    try:
        if pd.isna(raw):
            return []
    except (TypeError, ValueError):
        pass
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    found = []
    for token in (
        text.replace("[", " ")
        .replace("]", " ")
        .replace("'", " ")
        .replace('"', " ")
        .replace(",", " ")
        .split()
    ):
        if token.startswith("/m/") or token.startswith("/g/"):
            found.append(token)
    return found


def map_audioset_labels(mids: list[str]) -> tuple[str, bool, str]:
    mapped: list[str] = []
    for mid in mids:
        label = AUDIOSET_EV_MIDS.get(mid)
        if label and label not in mapped:
            mapped.append(label)
    multi = len(mapped) > 1
    if not mapped:
        if any(mid in AUDIOSET_EV_GENERIC_MIDS for mid in mids):
            type_label = "siren_untyped"
        else:
            type_label = "unknown"
    elif multi:
        type_label = "multi"
    else:
        type_label = mapped[0]
    return type_label, multi, "|".join(mapped)


def _yt_id_from_stem(stem: str) -> str:
    if stem.startswith("Y") and len(stem) > 1:
        return stem[1:]
    return stem


def _audioset_segment_from_path(path: Path) -> str | None:
    parts = [p.lower() for p in path.parts]
    for key in ("unbalanced", "balanced_train", "eval"):
        if key in parts:
            return key
    return None


def _audioset_polarity_from_path(path: Path) -> str | None:
    blob = "/".join(p.lower() for p in path.parts)
    if "negative_files" in blob or "/negatives/" in blob:
        return "negative"
    if "positive_files" in blob or "/positives/" in blob:
        return "positive"
    return None


def _truthy_downloaded(val) -> bool:
    if val is True:
        return True
    if val is False or val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        pass
    return str(val).strip().lower() in {"true", "1", "yes"}


def _read_audioset_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def scan_audioset_ev(root: Path) -> pd.DataFrame:
    pos_csv = None
    neg_csv = None
    for csv_path in root.rglob("*.csv"):
        name = csv_path.name.lower()
        if name == "ev_positives.csv":
            pos_csv = csv_path
        elif name == "ev_negatives.csv":
            neg_csv = csv_path

    meta_rows = []
    if pos_csv is not None:
        pos = _read_audioset_csv(pos_csv)
        if "downloaded" in pos.columns:
            pos = pos[pos["downloaded"].map(_truthy_downloaded)]
        for _, row in pos.iterrows():
            yt_id = str(row.get("yt_id", row.iloc[0])).strip()
            mids = parse_audioset_mids(row.get("positive_labels", row.get("labels")))
            type_label, multi, joined = map_audioset_labels(mids)
            meta_rows.append(
                {
                    "yt_id": yt_id,
                    "polarity": "positive",
                    "segment_type": str(row.get("segment_type", "")).strip() or None,
                    "downloaded_flag": row.get("downloaded"),
                    "type_label": type_label,
                    "multi_positive": multi,
                    "ev_labels": joined,
                    "mids": "|".join(mids),
                }
            )
    if neg_csv is not None:
        neg = _read_audioset_csv(neg_csv)
        if "downloaded" in neg.columns:
            neg = neg[neg["downloaded"].map(_truthy_downloaded)]
        for _, row in neg.iterrows():
            yt_id = str(row.get("yt_id", row.iloc[0])).strip()
            meta_rows.append(
                {
                    "yt_id": yt_id,
                    "polarity": "negative",
                    "segment_type": str(row.get("segment_type", "")).strip() or None,
                    "downloaded_flag": row.get("downloaded"),
                    "type_label": "urban_negative",
                    "multi_positive": False,
                    "ev_labels": "",
                    "mids": "",
                }
            )
    meta = pd.DataFrame(meta_rows)
    meta_by_id = {}
    if not meta.empty:
        meta_by_id = {str(r["yt_id"]): r for r in meta.to_dict(orient="records")}

    wav_rows = []
    for path in root.rglob("*"):
        if not path.is_file() or not is_audio(path):
            continue
        yt_id = _yt_id_from_stem(path.stem)
        info = meta_by_id.get(yt_id, {})
        polarity = info.get("polarity") or _audioset_polarity_from_path(path) or "unknown"
        segment = info.get("segment_type") or _audioset_segment_from_path(path) or "unknown"
        if polarity == "negative":
            type_label = "urban_negative"
            multi = False
            ev_labels = ""
            task = "binary"
        else:
            type_label = info.get("type_label") or "unknown"
            multi = bool(info.get("multi_positive", False))
            ev_labels = info.get("ev_labels") or ""
            task = "multiclass"
        wav_rows.append(
            {
                "corpus": "audioset_ev",
                "path": str(path),
                "relpath": str(path.relative_to(root)),
                "label": type_label,
                "task": task,
                "yt_id": yt_id,
                "group_id": yt_id,
                "polarity": polarity,
                "segment_type": segment,
                "multi_positive": multi,
                "ev_labels": ev_labels,
                "has_wav": True,
            }
        )

    wav_df = pd.DataFrame(wav_rows)
    if meta.empty:
        return wav_df

    seen = set(wav_df["yt_id"].astype(str)) if not wav_df.empty else set()
    missing_rows = []
    for rec in meta.to_dict(orient="records"):
        if str(rec["yt_id"]) in seen:
            continue
        polarity = rec["polarity"]
        missing_rows.append(
            {
                "corpus": "audioset_ev",
                "path": "",
                "relpath": "",
                "label": rec["type_label"],
                "task": "binary" if polarity == "negative" else "multiclass",
                "yt_id": rec["yt_id"],
                "group_id": rec["yt_id"],
                "polarity": polarity,
                "segment_type": rec["segment_type"] or "unknown",
                "multi_positive": rec["multi_positive"],
                "ev_labels": rec["ev_labels"],
                "has_wav": False,
            }
        )
    if missing_rows:
        wav_df = pd.concat([wav_df, pd.DataFrame(missing_rows)], ignore_index=True)
    return wav_df


def assign_audioset_protocol_split(polarity: str, segment_type: str) -> str:
    pol = (polarity or "").strip().lower()
    seg = (segment_type or "").strip().lower()
    if seg == "eval":
        return "test"
    if pol == "positive" and seg in {"unbalanced", "unbalanced_train"}:
        return "train_pool"
    if pol == "negative" and seg == "balanced_train":
        return "train_pool"
    if pol == "positive" and seg == "balanced_train":
        return "balanced_train_ref"
    return "unused"

