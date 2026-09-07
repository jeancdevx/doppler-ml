# Notebooks Doppler

Convenciones para los notebooks de exploración. El entorno de ejecución previsto es **Kaggle**, con las mismas rutas resolubles en local.

## Estilo

- Un notebook = una pregunta (origen, EDA o características).
- Celdas cortas y reproducibles de arriba abajo, con semilla fija (`SEED = 7`).
- Figuras en `reports/figures/` (local) o `/kaggle/working/reports/figures/` (Kaggle).
- Tablas CSV en `reports/tables/`.
- No entrenar modelos aquí.

## Orden

1. `01_origen_y_adquisicion.ipynb`
2. `02_exploracion.ipynb`
3. `03_variables_relevantes.ipynb`

## Datos

Ver [data/README.md](../data/README.md). El helper `src/paths.py` busca:

| Corpus | Local | Kaggle input slug |
| --- | --- | --- |
| sireNNet | `data/raw/sirennet` | `sirennet` |
| LSSiren | `data/raw/lssiren` | `lssiren` |
| UrbanSound8K | `data/raw/urbansound8k` | `urbansound8k` |

En Kaggle, añade los tres datasets al Kernel y ejecuta los notebooks en orden.
