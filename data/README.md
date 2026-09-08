# Corpus Doppler — montaje local y Kaggle

Kaggle es el entorno de desarrollo. Los WAV no se versionan en git (`data/raw/` está en `.gitignore`).

## Corpus de entrenamiento

**AudioSet-EV v2** (Zenodo [10.5281/zenodo.18668076](https://doi.org/10.5281/zenodo.18668076)).

```bash
# ZIP ~16 GB (WAV ~18–28 GB extraídos). No entra en --corpus all.
python scripts/download_corpora.py --corpus audioset_ev

# Solo CSV (si el ZIP de Zenodo falla): el notebook 01 también acepta
# EV_Positives.csv / EV_Negatives.csv ya colocados en data/raw/audioset_ev/
python scripts/download_corpora.py --corpus audioset_ev --audioset-metadata-only
python scripts/download_corpora.py --corpus audioset_ev --audioset-preview
```

## Piloto y tests de otro dominio

```bash
# WAV de sireNNet (~826 MB) — piloto / test limpio opcional, no entra al fit
python scripts/download_corpora.py --corpus sirennet

# Tablas de características de LSSiren (sin ~2.3 GB de WAV)
python scripts/download_corpora.py --corpus lssiren --lssiren-csv-only

# Metadatos de UrbanSound8K (sin ~5.6 GB de audio)
python scripts/download_corpora.py --corpus urbansound8k --urbansound-metadata-only
```

LSSiren WAV y UrbanSound8K audio se bajan cuando se evalúe el modelo fuera de YouTube.

## Kaggle

Los notebooks no importan el paquete `src`. La primera celda solo fija rutas:

- local: `data/raw/<corpus>`
- Kaggle: `/kaggle/input/datasets/<user>/<slug>/<slug>` (p. ej. `/kaggle/input/datasets/jeancdevx/audioset-ev/audioset_ev`), y también `/kaggle/input/<slug>`
- slugs: `audioset-ev` / `audioset_ev`, `sirennet`, `lssiren`, `urbansound8k`
- salidas: `reports/` en local, `/kaggle/working/reports/` en Kaggle

1. Añade AudioSet-EV como Input (y el piloto si lo usas).
2. Sube los `.ipynb` del repo.
3. Ejecuta de arriba abajo.
