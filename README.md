# Reconocimiento acústico de sirenas de vehículos de emergencia

Prototipo de aprendizaje supervisado para clasificar segmentos de audio como **EMERGENCY** o **NON_EMERGENCY**. El proyecto compara tres modelos sobre características acústicas y dos arquitecturas sobre espectrogramas Log-Mel, con particiones comunes y trazabilidad hasta la grabación de origen.

GitHub conserva las fuentes y la documentación. **Las notebooks se ejecutan en Kaggle.** Las fuentes refactorizadas no contienen salidas anteriores; sus nuevos resultados están pendientes de ejecución y revisión.

## Problema y alcance

La tarea consiste en detectar evidencia acústica de una sirena. Una etiqueta positiva no identifica el tipo de vehículo ni certifica una emergencia real. La aplicación vehicular se plantea como una integración futura del componente de percepción.

Esta versión cubre preparación y entrenamiento de los cinco modelos. La evaluación predictiva externa, las perturbaciones acústicas y el prototipo controlado permanecen pendientes. La clasificación por vehículo, la localización espacial, la fusión con cámaras, V2X y el control de maniobras quedan fuera del experimento principal.

El fundamento y las delimitaciones se encuentran en [objetivo general](docs/2/2.3/2.3.1/objetivo-general.md), [alcance](docs/2/2.5/delimitacion-del-alcance.md) y [selección del modelo](docs/5/5.1/modelos-considerados.md).

## Corpus y etiquetas

| Corpus | Papel | Correspondencia binaria |
|---|---|---|
| AudioSet-EV v2 | Desarrollo | Listas curadas y carpetas `Positive_files` / `Negative_files`; se auditan contradicciones y se conserva `positive_labels`. |
| sireNNet | Desarrollo | `ambulance`, `firetruck` y `police` → EMERGENCY; `traffic` → NON_EMERGENCY. |
| A3Siren-Recordings | Desarrollo | `Siren` → EMERGENCY; `Noise` → NON_EMERGENCY. Los ocho canales de una toma permanecen agrupados por track. |
| LSSiren | Reserva predictiva externa | `Ambulance` → EMERGENCY; `Road` → NON_EMERGENCY. |

Las fuentes y versiones están documentadas en [adquisición](docs/3/3.2/3.2.1/localizacion-y-obtencion-de-las-fuentes.md): AudioSet-EV v2, versión 0.2.0 ([DOI](https://doi.org/10.5281/zenodo.18668076)); LSSiren, distribución asociada a Asif et al. (2022); sireNNet, versión 1 ([DOI](https://doi.org/10.17632/j4ydzzv4kb.1)); y A3Siren-Recordings, repositorio asociado a Cantarini et al. (2022).

Las etiquetas de archivo o track se heredan a las ventanas cuando no hay anotaciones temporales verificadas. Se trata de supervisión débil. SHA-256 controla copias binarias exactas; los identificadores de procedencia amplían las dependencias conocidas. La falta de padres de aumentos, especialmente en sireNNet, se registra como limitación.

## Notebooks

| Orden | Fuente | Entrada | Resultado de la ejecución |
|---|---|---|---|
| 00 | [Preparación](notebooks/00_preparacion_insumos.ipynb) | Cuatro corpus originales y su metadata | Paquete preparado común y reportes de integridad. |
| 01 | [Regresión logística](notebooks/01_regresion_logistica.ipynb) | 21 características | Pipeline con escalamiento ajustado por fold y regularización L2. |
| 02 | [Random Forest](notebooks/02_random_forest.ipynb) | 21 características | Ensamble de 300 árboles por candidato. |
| 03 | [SVM](notebooks/03_svm.ipynb) | 21 características | LinearSVC o Nyström RBF + LinearSVC; la variante queda registrada. |
| 04 | [CNN compacta](notebooks/04_cnn.ipynb) | Log-Mel | Red entrenada desde cero. |
| 05 | [ResNet18](notebooks/05_resnet18.ipynb) | Log-Mel | Transferencia desde ImageNet1K_V1 y ajuste fino. |

Cada notebook contiene su código necesario. Los modelos dependen del paquete de preparación, pero pueden ejecutarse en cualquier orden y no importan módulos del repositorio.

## Ejecución en Kaggle

1. Importa `00_preparacion_insumos.ipynb` en una sesión de CPU y adjunta los cuatro corpus originales. Conserva sus WAV, directorios de clase y CSV.
2. Revisa `CFG['datasets']` y las rutas montadas en `/kaggle/input`. Ejecuta todas las celdas en orden. Usa una `output_root` nueva si cambian fuentes o configuración respecto de una ejecución anterior.
3. Descarga `prepared_data_v3_completo.zip` desde `/kaggle/working` y publícalo como dataset de Kaggle para adjuntarlo a los cinco modelos. Conserva `prepared_data_v3_reportes.zip` para revisar la preparación.
4. Importa cada notebook de modelo y adjunta **esa misma versión** del paquete completo. Con un único paquete, la detección es automática. Si hay varios, configura `prepared_zip` o `prepared_root`, exclusivamente uno de ellos. El ZIP de reportes no sirve para entrenar.
5. Ejecuta regresión logística, Random Forest y SVM en CPU. Para CNN y ResNet18 activa GPU; se admiten hasta dos dispositivos. Los lotes globales predeterminados son 64 y 128, respectivamente. Las fuentes usan `loader_workers=0`.
6. Para inicializar ResNet18, habilita Internet o adjunta los pesos oficiales `resnet18-f37072fd.pth` y configura `pretrained_weights_path`. Una descarga fallida detiene el entrenamiento. El modelo final exportado se recarga sin descargar nuevamente ImageNet.
7. Conserva la notebook ejecutada, el ZIP de resultados y el ZIP de modelos de cada ejecución. Compara ejecuciones con el mismo manifiesto de preparación y hash de folds.

Las bibliotecas se toman del entorno Kaggle; las notebooks no instalan ni actualizan paquetes automáticamente. Cada ejecución registra Python, versiones de paquetes, CPU y dispositivos utilizados. La compatibilidad efectiva del entorno se confirma al ejecutar.

## Preparación y contrato de datos

| Parámetro | Convención |
|---|---|
| Audio preparado | 16 kHz; mono por WAV; normalización por pico a 0,99. |
| Segmentación | Ventanas completas de 3 s, salto de 1,5 s; se descartan clips cortos y remanentes incompletos. |
| STFT | FFT y ventana de 2048 muestras, Hann periódica, salto de 512, centrado y padding constante. |
| Características tabulares | RMS, ZCR, frecuencia dominante, centroide, ancho de banda, roll-off al 85 %, flujo, planitud y trece medias MFCC. |
| Log-Mel | 64 bandas entre 20 y 8000 Hz, escala y normalización Slaney, potencia, referencia al máximo del segmento, rango de 80 dB; `float32`, 64 × 94. |
| MFCC | DCT-II ortonormal sobre Mel en dB con referencia 1; órdenes 0–12, denominados `mfcc_1`–`mfcc_13`. |

El contrato `doppler-prepared-v3` almacena una matriz por partición:

```text
prepared_data/
├── PREPARATION_READY.json
├── config/
├── metadata/
│   ├── metadata_segments.csv
│   ├── representation_manifest.csv
│   └── representation_containers.csv
├── splits/{train,validation,generalization}.csv
├── features/{acoustic_features,mfcc_features}.csv
├── representations/logmel/{train,validation,generalization}.npy
└── reports/artifact_hashes.json
```

`segment_id` es la clave de unión. `representation_path` señala el contenedor y `representation_row` su fila desde cero. Los modelos profundos leen las matrices de desarrollo por memoria mapeada; no construyen otra copia completa ni abren la matriz externa. Los lectores conservan compatibilidad con paquetes v2 de matrices individuales.

El paquete completo contiene un estado `PREPARED`: datos validados y listos para consumo. El estado `COMPLETE` en la salida de preparación y el ZIP de reportes añade la confirmación de exportación. Los hashes cubren metadata, configuración, características y contenedores. La reanudación exige igualdad de fuentes, configuración y asignación de filas, y comprueba el contenido reutilizado.

Consulta el [contrato de preparación](docs/contrato-preparacion.md) para sus controles y compatibilidad.

## Evaluación y reproducibilidad

Las particiones fijas se asignan antes de segmentar, por grupos de origen, con semilla 42 y proporción objetivo de validación del 20 %. La búsqueda usa tres folds agrupados exclusivamente dentro de `train`. Los cinco modelos comparten esa asignación. Escaladores, normalización por banda y pesos de clase se ajustan dentro del subconjunto correspondiente.

La selección utiliza F1 macro medio; los desempates consideran recall y tiempo de ajuste. El umbral se elige sobre predicciones OOF buscando **recall ≥ 0,90 y FPR < 0,10**. Si ambas restricciones son inviables, se registra el fallback al máximo F1 macro OOF. La validación fija se evalúa después con parámetros y umbral congelados. Las redes reservan un subconjunto interno para elegir épocas y fijan las del ajuste final por la mediana obtenida en CV.

Se generan métricas globales y por corpus, predicciones, intervalos mediante bootstrap de grupos y costos por etapa. La latencia medida usa entradas ya representadas; no incluye la adquisición de tres segundos ni la extracción acústica. LSSiren no participa en selección, parada, normalización aprendida ni predicción en estas notebooks.

El registro de semillas y versiones permite examinar reproducibilidad. Las diferencias de hardware, bibliotecas y reducción entre GPU pueden cambiar redondeos; las verificaciones numéricas utilizan tolerancias documentadas. El cumplimiento de [criterios de éxito](docs/2/2.4/criterios-de-exito.md) requerirá la evaluación posterior correspondiente.

## Artefactos de entrenamiento

Cada ejecución crea una carpeta bajo `/kaggle/working/entrenamiento/<modelo>/<run_id>/`:

- `config/`: configuración, preparación de origen, contrato de entrada y experimento congelado.
- `tables/` y `predictions/`: folds, búsqueda, umbral, validación fija y tiempos.
- `figures/`: figuras PNG y SVG generadas durante la ejecución.
- `logs/`: registro legible, eventos JSONL y TensorBoard cuando esté instalado y habilitado.
- `models/`: pipeline tabular o pesos profundos, normalización y umbral.
- `reports/`: estado, hashes, limitaciones y comprobación de recarga profunda.

`*_resultados.zip` conserva evidencia sin pesos ni caché temporal. `*_modelos.zip` conserva modelos y configuración. Ante un error, `*_diagnostico.zip` reúne los registros disponibles. Los ZIP finales se publican después de verificar su CRC.

## Verificación de las fuentes

Los tests utilizan corpus WAV artificiales pequeños. Comprueban contrato compacto, compatibilidad v2, grupos, corrupción, selección de umbral, entrenamiento y recarga; sus salidas no son resultados experimentales del proyecto.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r tests/requirements.txt
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/ruff check notebooks tests --select F,E4,E7,E9
MPLBACKEND=Agg .venv/bin/python -m pytest tests -q
```

La ejecución completa con los corpus reales, GPU de Kaggle y pesos oficiales de transferencia permanece pendiente. La revisión local no acredita desempeño ni aceleración en ese entorno.

## Documentación

- [Documento del proyecto, puntos 1–5](docs/README.md).
- [Auditoría de las seis notebooks](docs/auditoria-notebooks.md).
- [Contrato compacto y compatibilidad](docs/contrato-preparacion.md).
