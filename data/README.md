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

1. Crear datasets (privados o públicos) a partir de `data/raw/sirennet`, `data/raw/lssiren` y `data/raw/urbansound8k`.
2. UrbanSound8K también puede añadirse desde el catálogo de Kaggle si ya existe una copia (`urbansound8k`).
3. En el Kernel: **Add data** con slugs `sirennet`, `lssiren`, `urbansound8k`.
4. Subir los tres notebooks. Los paths se resuelven solos vía `src/paths.py`.

Si Kaggle no monta el directorio `src/`, copia `paths.py` a la primera celda o añade el repo como dataset auxiliar.
