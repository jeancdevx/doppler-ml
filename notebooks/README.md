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

Ver [data/README.md](../data/README.md). El helper de la primera celda busca:

| Corpus | Local | Kaggle |
| --- | --- | --- |
| sireNNet | `data/raw/sirennet` | `/kaggle/input/datasets/<user>/sirennet/sirennet` o `/kaggle/input/sirennet` |
| LSSiren | `data/raw/lssiren` | `/kaggle/input/datasets/<user>/lssiren/lssiren` |
| UrbanSound8K | `data/raw/urbansound8k` | `/kaggle/input/datasets/<user>/urbansound8k/urbansound8k` |

En Kaggle, añade los tres datasets al Kernel (`sirennet`, `lssiren`, `urbansound8k`) y ejecuta los notebooks en orden. La primera celda elige `DATA_ROOT` (`data/raw` en local, `/kaggle/input` en Kaggle). No hace falta subir `src/`.
