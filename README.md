# Sistema de Reconocimiento Acústico de Vehículos de Emergencia para Vehículos Autónomos

Proyecto **Doppler**. El análisis de las secciones **3.2–3.5** (recolección y comprensión de datos) vive en una sola notebook, pensada para **Kaggle** (y Colab como fallback).

## Notebook

[`notebooks/recoleccion_y_comprension.ipynb`](notebooks/recoleccion_y_comprension.ipynb)

Ejecute de arriba abajo. Bloques:

0. Entorno e Inputs de Kaggle  
3.2 Proceso de adquisición  
3.3 Exploración de los datos  
3.4 Análisis de variables relevantes  
3.5 Hallazgos iniciales  

Al final de 3.2–3.5 hay celdas de **texto en español para pegar en el PDF**. Tablas y figuras salen a `reports/tables` y `reports/figures` (en Kaggle: `/kaggle/working/reports`).

## Kaggle: audio en Input, no en Working

`/kaggle/working` tiene ~20 GB. **No descargue los corpora ahí.**

1. Cree un *Kaggle Dataset* por fuente (**Add Input → New Dataset**) subiendo el ZIP/TAR **oficial**, o adjunte un dataset público ya existente.
2. En el notebook: **Add Input** y monte todos los corpora.
3. Active Internet solo si falta un Input o usa el fallback de Colab.
4. CPU basta para este EDA.

La notebook resuelve cada corpus por *slug* o por archivos marcadores (`UrbanSound8K.csv`, `EV_Positives.csv`, etc.). Si el Input llega comprimido y no hay WAV, extrae a `/kaggle/temp/doppler-data` (scratch de sesión). RecSIR demo (pocos clips) sí puede escribirse en `/kaggle/working/recsir_demo`.

### Slugs que busca la notebook

| Corpus | Slugs | Origen |
|---|---|---|
| LSSiren | `lssiren`, `large-scale-audio-dataset-for-emergency-vehicle` | [Figshare 19291472](https://doi.org/10.6084/m9.figshare.19291472) |
| sireNNet | `sirennet`, `j4ydzzv4kb` | [Mendeley j4ydzzv4kb](https://data.mendeley.com/datasets/j4ydzzv4kb/1) |
| AudioSet-EV v2 | `audioset-ev-v2`, `audioset-ev` | [Zenodo 18668076](https://zenodo.org/records/18668076) |
| UrbanSound8K | `urbansound8k` | [Zenodo 1203745](https://zenodo.org/records/1203745) o dataset público de Kaggle |
| SONYC-UST v2 | `sonyc-ust-v2`, `sonyc-ust` | [Zenodo 3966543](https://zenodo.org/records/3966543) |
| ESC-50 | `esc50`, `esc-50` | GitHub ESC-50 o dataset público |

UrbanSound8K y ESC-50 ya existen como datasets públicos de Kaggle: adjúntelos en lugar de re-subirlos.

RecSIR/SynSIR **no tienen ZIP público**; la notebook genera una malla Doppler×SNR. FSD50K (~31 GB) no se monta por defecto.

## Colab

Si no hay `/kaggle/input`, ponga `DOWNLOAD_MISSING = True` (es el valor por defecto fuera de Kaggle). Las celdas descargan a `/content/doppler-data` con `wget -c` y MD5 cuando existe. Baje AudioSet-EV primero (pico zip+extract ~44 GB).

## Flags

En la primera celda de código:

- `SMOKE = False` — informe completo. `True` usa 1–pocos archivos por corpus para validar la tubería.
- `DOWNLOAD_MISSING` — `False` en Kaggle (Inputs); `True` en Colab.
- `SAMPLE_PER_CORPUS` — tamaño de la muestra de descriptores (200 por defecto).

## Dependencias

Ver [`requirements.txt`](requirements.txt). En Kaggle/Colab la notebook instala lo que falte (`librosa`, `soundfile`, `seaborn`, `tqdm`, y `pyroadacoustics` si está disponible).
