# Corpus Doppler — montaje local y Kaggle

Kaggle es el entorno de desarrollo. Los WAV no se versionan en git (`data/raw/` está en `.gitignore`).

## Corpus de trabajo (EDA)

| Corpus | Fuente | Licencia | Cómo obtenerlo |
| --- | --- | --- | --- |
## Cómo obtenerlo

```bash
# WAV de sireNNet (~826 MB) — corpus primario
python scripts/download_corpora.py --corpus sirennet

# Tablas de características de LSSiren (sin ~2.3 GB de WAV)
python scripts/download_corpora.py --corpus lssiren --lssiren-csv-only

# WAV completos de LSSiren
python scripts/download_corpora.py --corpus lssiren

# Metadatos de UrbanSound8K (sin ~5.6 GB de audio)
python scripts/download_corpora.py --corpus urbansound8k --urbansound-metadata-only

# Audio completo de UrbanSound8K
python scripts/download_corpora.py --corpus urbansound8k
```

## Corpus de escala (entrenamiento posterior)

AudioSet-EV v2 ([Zenodo 10.5281/zenodo.18668076](https://doi.org/10.5281/zenodo.18668076)): ~8–16 GB zip / ~28 GB WAV. No se descarga en esta fase; se documenta en el notebook 01.

## Kaggle

Los notebooks no importan el paquete `src`. La primera celda solo fija rutas:

- local: `data/raw/<corpus>`
- Kaggle: `/kaggle/input/datasets/<user>/<slug>/<slug>` (p. ej. `/kaggle/input/datasets/jeancdevx/sirennet/sirennet`), y también `/kaggle/input/<slug>` si el Input es plano
- salidas: `reports/` en local, `/kaggle/working/reports/` en Kaggle

1. Añade esos tres datasets como Input.
2. Sube de nuevo los `.ipynb` del repo (tira la copia con el zip en base64).
3. Ejecuta de arriba abajo.
