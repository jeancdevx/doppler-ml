# Notebooks Doppler

Convenciones para los notebooks. El entorno de ejecución previsto es **Kaggle**, con las mismas rutas resolubles en local.

## Estilo

- Un notebook = una pregunta (origen, EDA, características o preparación).
- Celdas cortas y reproducibles de arriba abajo, con semilla fija (`SEED = 7`).
- Figuras en `reports/figures/` (local) o `/kaggle/working/reports/figures/` (Kaggle).
- Tablas CSV en `reports/tables/`.
- No entrenar modelos aquí.

## Orden

1. `01_origen_y_adquisicion.ipynb`
2. `02_exploracion.ipynb`
3. `03_variables_relevantes.ipynb`
4. `04_preparacion.ipynb`

## Datos

Ver [data/README.md](../data/README.md). El helper de la primera celda busca:

| Corpus | Local | Kaggle |
| --- | --- | --- |
| AudioSet-EV | `data/raw/audioset_ev` | `/kaggle/input/datasets/<user>/audioset-ev/audioset_ev` o `/kaggle/input/audioset-ev` |
| sireNNet (piloto) | `data/raw/sirennet` | `/kaggle/input/datasets/<user>/sirennet/sirennet` |
| LSSiren | `data/raw/lssiren` | `/kaggle/input/datasets/<user>/lssiren/lssiren` |
| UrbanSound8K | `data/raw/urbansound8k` | `/kaggle/input/datasets/<user>/urbansound8k/urbansound8k` |

En Kaggle, añade al menos **audioset-ev** al Kernel. La primera celda elige `DATA_ROOT`. No hace falta subir `src/`.
