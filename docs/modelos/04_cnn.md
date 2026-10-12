# CNN sobre Log-Mel: documentación del modelo entrenado

Este documento describe la ejecución entregada del **12 de octubre de 2026**, desde el audio de origen hasta el artefacto final. Explica el procedimiento que realmente ejecutó la notebook, los parámetros guardados y las métricas exportadas. Las fórmulas desarrollan esas operaciones; las interpretaciones se distinguen de las mediciones. Se puede leer de manera independiente de los informes de los otros modelos.

El modelo obtuvo **F1 macro = 0,892269**, **recall de emergencia = 0,792430** y **FPR = 0,035461** en 29 315 segmentos de validación interna fija. El umbral exportado es **`0.5282900333404541`**. La búsqueda OOF no encontró un umbral que satisficiera simultáneamente recall ≥ 0,90 y FPR < 0,10; se utilizó la alternativa prevista de máximo F1 macro OOF. El recall por segmento de la validación fija tampoco alcanza 0,90. **La evaluación externa de LSSiren sigue pendiente.**

## Guía de lectura

- [1. Identidad de la ejecución y fuentes](#seccion-1)
- [2. Corpus, etiquetas y limpieza previa](#seccion-2)
- [3. Preprocesamiento de la señal](#seccion-3)
- [4. Entrada Log-Mel, almacenamiento y normalización](#seccion-4)
- [5. Particiones, grupos y folds](#seccion-5)
- [6. Arquitectura y matemática de la CNN](#seccion-6)
- [7. Entrenamiento profundo y parada interna](#seccion-7)
- [8. Búsqueda de candidatos y ajuste seleccionado](#seccion-8)
- [9. Umbral OOF y restricciones operativas](#seccion-9)
- [10. Resultados de la validación interna fija](#seccion-10)
- [11. Figuras originales y lectura técnica](#seccion-11)
- [12. Entorno, tiempos y costo computacional](#seccion-12)
- [13. Artefacto final, trazabilidad y recarga](#seccion-13)
- [14. Alcance de las conclusiones y trabajo pendiente](#seccion-14)

<a id="seccion-1"></a>

## 1. Identidad de la ejecución y fuentes

| Campo | Valor |
| --- | --- |
| Notebook | [04_cnn.ipynb](../../notebooks/04_cnn.ipynb) |
| Modelo | `04_cnn` |
| Estado de entrenamiento | `COMPLETE` |
| Inicio UTC | `2026-10-12T16:50:48.589301+00:00` |
| Cierre de entrenamiento UTC | `2026-10-12T17:31:21.245627+00:00` |
| Semilla de preparación y entrenamiento | 42 |
| Evaluación externa | `external_evaluated=false`; `holdout_evaluated=false` |
| Resultados originales | `data/results/04_cnn_resultados/` |
| Modelo original | `data/models/04_cnn_modelos/models/` |
| Configuración efectiva | [config/training_config.json](evidencia/04_cnn/config/training_config.json) |
| Experimento congelado | [config/frozen_experiment.json](evidencia/04_cnn/config/frozen_experiment.json) |
| Estado exportado | [reports/run_status.json](evidencia/04_cnn/reports/run_status.json) |

La clase positiva es `EMERGENCY=1`; la negativa, `NON_EMERGENCY=0`. El problema es clasificación binaria de ventanas: no distingue ambulancia, bomberos y policía como salidas separadas, no estima distancia ni dirección y no localiza el inicio exacto de un evento. El marco del proyecto se encuentra en [objetivos](../2/2.3/objetivos.md), [alcance](../2/2.5/delimitacion-del-alcance.md) y [modelos considerados](../5/5.1/modelos-considerados.md).

Las tablas y figuras enlazadas en `evidencia/04_cnn/` y `figuras/04_cnn/` son copias de las exportaciones, con igualdad de SHA-256 comprobada. Los archivos de `inspeccion/` son derivados identificados: lectura de parámetros y recálculo de métricas existentes, sin volver a entrenar. [El manifiesto](evidencia/manifest.csv) permite distinguir ambos orígenes. Los archivos binarios permanecen en `data/models`; no se incorporan a la documentación.

<a id="seccion-2"></a>

## 2. Corpus, etiquetas y limpieza previa

### 2.1. Origen de la etiqueta binaria

| Corpus | Etiqueta positiva | Etiqueta negativa | Participación |
| --- | --- | --- | --- |
| AudioSet-EV v2 | Rutas documentadas `Positive_files` / `EV_Positives`, contrastadas con metadata | `Negative_files` / `EV_Negatives` | Entrenamiento y validación fija |
| sireNNet | `ambulance`, `firetruck`, `police` | `traffic` | Entrenamiento y validación fija |
| A3Siren-Recordings | `Siren` | `Noise` | Entrenamiento y validación fija, agrupando los canales de cada track |
| LSSiren | `Ambulance` | `Road` | Reserva exclusiva para generalización externa |

Se conserva la procedencia y, en AudioSet, las etiquetas originales cuando están disponibles. La etiqueta de archivo o track se hereda a cada ventana. Una ventana positiva puede contener intervalos sin sirena o ruido dominante: las etiquetas son débiles respecto de la actividad exacta dentro de esos tres segundos. No se introdujo una nueva anotación temporal ni aumento de datos en esta ejecución (`new_augmentation=false`). Las versiones aumentadas que pudiera traer sireNNet pertenecen al corpus de origen.

### 2.2. Qué se excluyó realmente

El inventario inicial contiene **32 429 archivos**. Se excluyeron lógicamente dos audios inválidos de AudioSet-EV v2 y 169 duplicados exactos: 20 de AudioSet, 141 de LSSiren y 8 de sireNNet. Quedaron 32 258 archivos después de esta limpieza. A continuación se descartaron 54 archivos demasiado cortos para una ventana completa: 21 de AudioSet y 33 de LSSiren. La preparación final usa **32 204 archivos** y genera **153 993 segmentos**, incluyendo la reserva externa.

| Corpus | Conservados tras limpieza | Duplicados excluidos | Inválidos excluidos | Cortos excluidos después |
| --- | --- | --- | --- | --- |
| AudioSet-EV v2 | 28 794 | 20 | 2 | 21 |
| sireNNet | 1 667 | 8 | 0 | 0 |
| A3Siren-Recordings | 104 | 0 | 0 | 0 |
| LSSiren | 1 693 | 141 | 0 | 33 |

La deduplicación usa SHA-256 de archivos: detecta igualdad binaria, pero no audios iguales que hayan sido recodificados o modificados. Las alertas de silencio, clipping y posible cancelación al promediar canales se registran; su existencia no equivale a aplicar un filtro nuevo de eliminación. Evidencia: [limpieza](evidencia/preparacion/reports/cleaning_summary.csv), [validaciones finales](evidencia/preparacion/reports/final_preparation_validation.csv) y [alcance de independencia](evidencia/preparacion/reports/independence_scope.json).

<a id="seccion-3"></a>

## 3. Preprocesamiento de la señal

### 3.1. Decodificación, canales, remuestreo y amplitud

El audio se decodifica a valores numéricos, se promedian los canales que contiene cada WAV y se convierte a `float32`. Para un WAV con $C$ canales:

$$
x_{mono}[n]=\frac{1}{C}\sum_{c=1}^C x_c[n].
$$

Los ocho WAV de canal de un track A3 se conservan como fuentes separadas. El promedio anterior se aplica **dentro de cada WAV**, no entre esos ocho archivos. Todos los canales y ventanas de un track comparten grupo para impedir que se distribuyan entre entrenamiento y validación.

Si la tasa original difiere, `librosa.resample(..., res_type="soxr_hq")` lleva la señal a **16 000 Hz**. Después se normaliza el pico de **la grabación completa**, antes de segmentar:

$$
a=\max_n|x_{mono}[n]|,\qquad
\widetilde x[n]=\begin{cases}0,99\,x_{mono}[n]/a,&a>0,\\x_{mono}[n],&a=0.\end{cases}
$$

El factor se calcula después del remuestreo. Esta operación homogeniza el máximo digital de cada grabación, pero no recupera el nivel de presión sonora original, no elimina distorsión por clipping y puede amplificar una señal con pico muy pequeño. El RMS posterior expresa energía relativa dentro de una grabación normalizada, no decibelios SPL calibrados. Como el pico usa toda la grabación, una implementación en flujo continuo necesitaría resolver explícitamente esta dependencia; no se ha medido aquí una normalización causal en línea.

### 3.2. Ventanas y dependencia temporal

Se extraen **48 000 muestras** por ventana (3 s), con salto de **24 000 muestras** (1,5 s): solapamiento del 50 %. Para una señal de $L$ muestras:

$$
N_{ventanas}=\begin{cases}1+\left\lfloor\dfrac{L-48000}{24000}\right\rfloor,&L\ge48000,\\0,&L<48000.\end{cases}
$$

La ventana $i$ comienza en $i\,24000$ y termina antes de $i\,24000+48000$. Se descartan clips cortos y colas incompletas; no hay relleno para producir ventanas de audio adicionales. El relleno de borde de la STFT que se explica enseguida es una operación distinta. Los segmentos solapados están correlacionados y no deben tratarse como grabaciones independientes. Multiplicar el número de ventanas por 3 s cuenta audio repetido por solapamiento, no duración única de captura.

### 3.3. STFT reutilizada para las representaciones

Para cada segmento se calcula una única STFT con ventana Hann periódica de 2 048 muestras, salto de 512, `center=True` y relleno constante cero. Si $x_p$ denota la señal con 1 024 ceros añadidos a cada borde:

$$
D[k,t]=\sum_{n=0}^{2047}x_p[t\,512+n]h[n]e^{-j2\pi kn/2048},\qquad
h[n]=\tfrac12\left(1-\cos\tfrac{2\pi n}{2048}\right).
$$

Se conservan $k=0,\dots,1024$, con $f_k=k\,16000/2048$. La separación entre bins es **7,8125 Hz**, la ventana de análisis dura **128 ms** y los centros de frames están separados **32 ms**. Hay $1+\lfloor48000/512\rfloor=94$ frames por segmento: magnitud y potencia tienen dimensión **1 025 × 94**. Se define $A[k,t]=|D[k,t]|$ y $P[k,t]=A[k,t]^2$. La resolución efectiva también depende de la ventana; la separación entre bins no garantiza distinguir dos tonos separados exactamente por esa cantidad. Estas dimensiones se derivan del código ejecutado y de la [convención de STFT de Librosa 0.11.0](https://librosa.org/doc/0.11.0/generated/librosa.stft.html).

### 3.4. Mel, decibelios y diferencia entre MFCC y Log-Mel

Un banco $B\in\mathbb R^{64\times1025}$ proyecta la potencia a 64 bandas entre 20 y 8 000 Hz:

$$
M[m,t]=\sum_{k=0}^{1024}B[m,k]P[k,t].
$$

El banco usa escala Slaney (`htk=False`) y normalización por área (`norm="slaney"`). Los triángulos están distribuidos según esa escala; no se aplica la fórmula HTK $2595\log_{10}(1+f/700)$ como sustitución. [La API de Mel](https://librosa.org/doc/0.11.0/generated/librosa.filters.mel.html) identifica estas opciones.

Para las redes, el Log-Mel exportado usa $R=\max_{m,t}M[m,t]$, piso $\epsilon=10^{-10}$ y rango de 80 dB:

$$
L_0[m,t]=10\log_{10}\max(M[m,t],\epsilon)-10\log_{10}\max(R,\epsilon),
$$

$$
L[m,t]=\max\left(L_0[m,t],\max_{m,t}L_0[m,t]-80\right).
$$

Así se obtiene una matriz **64 × 94 `float32`**, con máximo relativo 0 dB y piso de hasta −80 dB. La referencia al máximo corresponde al segmento completo; elimina el nivel absoluto de su espectrograma. Una señal enteramente nula produce un caso degenerado con la referencia limitada por el mismo piso, no una medición de intensidad física.

Los MFCC reutilizan $M$, pero transforman a dB con **`ref=1.0`**, no con `ref=max`. Sobre esa matriz de dB, también limitada a 80 dB respecto de su máximo, se aplica DCT-II ortonormal sin lifter y se promedian 13 coeficientes en el tiempo. Confundir ambas referencias cambia especialmente el coeficiente de orden cero y deja de reproducir los insumos empleados.
<a id="seccion-4"></a>

## 4. Entrada Log-Mel, almacenamiento y normalización

La red recibe $L\in\mathbb R^{64\times94}$ por segmento. Las 21 características tabulares también fueron preparadas, pero **no se concatenan ni entran a esta arquitectura**. El tensor conserva bandas de frecuencia y evolución temporal de cada ventana; no es una imagen RGB del audio ni una figura renderizada con un mapa de colores.

### 4.1. Contrato compacto y acceso por filas

La ejecución consume `doppler-prepared-v3`. Cada partición tiene un contenedor `.npy`: train **(118 314, 64, 94)**, validation **(29 315, 64, 94)** y generalization **(6 364, 64, 94)**. Sus cargas numéricas ocupan aproximadamente 2,652 GiB, 0,657 GiB y 0,143 GiB, respectivamente, sin contar cabeceras. La red abre solo los contenedores de desarrollo para su consumo predictivo.

`segment_id` conserva la unión lógica; `representation_path` identifica el contenedor y `representation_row` señala una fila con índice desde cero. La notebook contrasta hashes publicados, dimensiones, finitud y asignación de filas, y rechaza índices repetidos o inválidos. Los lectores mantienen compatibilidad con v2, pero esta ejecución utiliza v3.

Las matrices se abren con `numpy.load(..., mmap_mode="r", allow_pickle=False)`. El mapeo permite recuperar filas sin construir otra matriz completa de desarrollo. Las selecciones por lote y la normalización sí crean buffers temporales. Además, las páginas mapeadas que el sistema mantiene residentes pueden aumentar el RSS; memoria mapeada no significa RAM constante ni ausencia de costo de lectura. [Referencia de NumPy](https://numpy.org/doc/stable/reference/generated/numpy.load.html).

### 4.2. Estadísticas por banda, ajustadas donde corresponde

Para las filas de ajuste $\mathcal F$, con $n_F$ segmentos y $T=94$, se estima por cada banda $m$:

$$
\mu_m=\frac1{n_FT}\sum_{i\in\mathcal F}\sum_t L_i[m,t],\qquad
\sigma_m=\sqrt{\max\left(\frac1{n_FT}\sum_{i\in\mathcal F}\sum_tL_i[m,t]^2-\mu_m^2,10^{-12}\right)}.
$$

$$
Z_i[m,t]=\frac{L_i[m,t]-\mu_m}{\sigma_m}.
$$

Las sumas se acumulan en `float64`, en bloques de 256 segmentos; los 64 valores de media y desviación resultantes se guardan en `float32`. No se normaliza cada posición temporal de forma independiente ni se fuerza cada espectrograma a media cero. El piso de varianza es $10^{-12}$; corresponde a una desviación mínima de $10^{-6}$.

Durante CV, $\mathcal F$ incluye solo el subconjunto que recibe gradientes, excluyendo tanto la parada interna como la validación del fold externo. Para el ajuste final, incluye todos los 118 314 segmentos de train y excluye los 29 315 de validación fija. Las estadísticas finales de CNN y ResNet18 coinciden porque utilizan las mismas filas y representación. Se guardan junto a los pesos: recalcularlas con datos de producción cambia el modelo efectivo.

### 4.3. DataLoader y tensores efectivos

Después de normalizar se añade el canal: un ejemplo es **(1, 64, 94)**. La lectura vectorizada por lote evita abrir un archivo por cada segmento. El `DataLoader` usa `num_workers=0`, `drop_last=False`, generador con semilla 42 y `shuffle=True` durante ajuste; las evaluaciones conservan orden. En CUDA usa `pin_memory=True` y transferencias con `non_blocking=True`. La última tanda puede contener menos ejemplos que el lote configurado.

Contrato consumido: [config/representation_contract.json](evidencia/04_cnn/config/representation_contract.json). Estadísticas de las 64 bandas: [inspeccion/normalization.csv](evidencia/04_cnn/inspeccion/normalization.csv). El artefacto [models/inference.json](evidencia/04_cnn/models/inference.json) guarda forma, umbral y normalización. Los hashes de los contenedores se encuentran en [el manifiesto de preparación](evidencia/preparacion/metadata/representation_containers.csv).

<a id="seccion-5"></a>

## 5. Particiones, grupos y folds

### 5.1. Separación fija antes de segmentar

Se reserva LSSiren completo como `generalization`. Los otros tres corpus se dividen por grupos, antes de crear ventanas. Para cada estrato de corpus y composición de etiquetas del grupo, se ordenan identificadores, se barajan con `numpy.random.default_rng(42)` y se asigna a validación un número de grupos cercano al 20 %, redondeado y limitado para conservar grupos en ambos lados cuando es posible. Esto no exige exactamente un 20 % de segmentos: la duración de las grabaciones y el número de canales varían.

| Partición | Corpus | Emergencia | No emergencia | Segmentos | Archivos | Grupos conocidos |
| --- | --- | --- | --- | --- | --- | --- |
| generalization | LSSiren | 2 888 | 3 476 | 6 364 | 1 660 | 1 660 |
| train | A3Siren-Recordings | 944 | 1 560 | 2 504 | 88 | 11 |
| train | AudioSet-EV v2 | 31 333 | 83 143 | 114 476 | 23 018 | 23 018 |
| train | sireNNet | 997 | 337 | 1 334 | 1 334 | 1 334 |
| validation | A3Siren-Recordings | 40 | 312 | 352 | 16 | 2 |
| validation | AudioSet-EV v2 | 7 848 | 20 782 | 28 630 | 5 755 | 5 755 |
| validation | sireNNet | 249 | 84 | 333 | 333 | 333 |

Totales: **train = 118 314 segmentos / 24 440 archivos / 24 363 grupos**, **validation = 29 315 / 6 104 / 6 090**, **generalization = 6 364 / 1 660 / 1 660**. Train tiene 33 274 positivos y 85 040 negativos; validación fija tiene 8 137 positivos y 21 178 negativos. LSSiren conserva 2 888 positivos y 3 476 negativos únicamente como reserva.

La proporción de emergencia en train es 28,123 %; en validación, 27,757 %. Las cifras corresponden a esta ejecución v3. Los conteos de documentos anteriores —por ejemplo, 118 349 y 29 280— pertenecen a evidencia histórica y no describen estas particiones.

### 5.2. Qué significa un grupo y qué independencia se comprobó

Se preservan los grupos declarados en el paquete. Las relaciones de archivo, track, hash y columnas recuperables `source_video_id`, `original_recording_id`, `augmentation_parent_id` se unen en componentes de dependencia, en vez de sustituir silenciosamente los grupos originales. Una relación conocida que cruce particiones provoca rechazo. Los identificadores de grupo se usan para dividir datos, no como entrada del predictor.

El reporte de entrenamiento registra **cero grupos conocidos cruzando splits**. En A3, 104 archivos pertenecen a 13 tracks: train tiene 11 tracks (6 positivos y 5 negativos) y validación solo 2 (uno positivo y uno negativo). Los ocho canales de cada track están juntos. La agrupación evita fuga entre estos canales, aunque cada canal sigue contribuyendo como un ejemplo distinto al entrenamiento y a la métrica por segmento.

Hay **1 667 archivos de sireNNet sin metadata de padre original o de aumento**. Por ello, la separación garantiza las dependencias identificadas, pero no demuestra independencia de todas las versiones aumentadas aguas arriba. Tampoco se realizó búsqueda de duplicados perceptuales. Una cifra grande de segmentos no elimina estas limitaciones. Evidencia: [tables/leakage_audit.csv](evidencia/04_cnn/tables/leakage_audit.csv) y [reports/independence_scope.json](evidencia/04_cnn/reports/independence_scope.json).

### 5.3. Validación cruzada de selección

Solo train se subdivide con `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)`. La estratificación de CV se hace sobre la etiqueta binaria, manteniendo cada grupo indivisible; no se estratifica conjuntamente por corpus y clase. La validación fija no se usa para elegir candidatos ni umbral.

| Fold, índice desde cero | Train del fold externo | Validación del fold externo | Positivos externos | Negativos externos | Grupos externos |
| --- | --- | --- | --- | --- | --- |
| 0 | 79 114 | 39 200 | 11 052 | 28 148 | 8 099 |
| 1 | 78 303 | 40 011 | 11 065 | 28 946 | 8 101 |
| 2 | 79 211 | 39 103 | 11 157 | 27 946 | 8 163 |

Cada uno de los 118 314 segmentos aparece una vez en validación externa de CV, generando una puntuación OOF (*out of fold*). Las cinco notebooks comparten exactamente asignaciones de folds e identificadores de validación fija. El SHA-256 registrado para las asignaciones es:

```text
3528557381804ea80cf8a9fd230bbe0c37f24e721094162609a2c41f83a2154d
```

La distribución de corpus no queda equilibrada automáticamente: el fold 2 contiene 320 segmentos positivos A3 y **ningún segmento negativo A3**. Ambos valores de la etiqueta sí existen globalmente en cada fold. Por ello, los resultados globales de CV no equivalen a una estimación estable por corpus, particularmente en A3. El detalle completo se conserva en [tables/cv_distribution.csv](evidencia/04_cnn/tables/cv_distribution.csv).

Las métricas de CV y el umbral OOF forman parte de la selección sobre desarrollo. Como se eligió el candidato usando estos folds, su puntuación OOF no es una estimación completamente independiente de esa selección. El procedimiento no implementa una evaluación externa anidada que repita toda la selección. La validación fija aporta otra partición de desarrollo; la evidencia de generalización a otro corpus requiere LSSiren, todavía pendiente.

<a id="seccion-6"></a>

## 6. Arquitectura y matemática de la CNN

### 6.1. Arquitectura efectiva y dimensiones

Es una red pequeña entrenada **desde cero**, sin pesos preentrenados. Sus tres bloques repiten `Conv2d → GroupNorm → ReLU → MaxPool2d`. Las convoluciones usan kernel 3 × 3, stride 1, padding 1 y `bias=False`; GroupNorm tiene cuatro grupos. El lote global es 64, repartido entre dos GPU en esta ejecución.

| Etapa | Operación | Forma de salida, sin dimensión de lote | Parámetros entrenables |
| --- | --- | --- | --- |
| Entrada | Log-Mel estandarizado | 1 × 64 × 94 | 0 |
| Bloque 1, convolución | 1 → 16, kernel 3 × 3 | 16 × 64 × 94 | 144 |
| Bloque 1, GroupNorm | 4 grupos, 16 canales | 16 × 64 × 94 | 32 |
| Bloque 1, pooling | ReLU; MaxPool 2 × 2, stride 2 | 16 × 32 × 47 | 0 |
| Bloque 2, convolución | 16 → 32, kernel 3 × 3 | 32 × 32 × 47 | 4 608 |
| Bloque 2, GroupNorm | 4 grupos, 32 canales | 32 × 32 × 47 | 64 |
| Bloque 2, pooling | ReLU; MaxPool 2 × 2, stride 2 | 32 × 16 × 23 | 0 |
| Bloque 3, convolución | 32 → 64, kernel 3 × 3 | 64 × 16 × 23 | 18 432 |
| Bloque 3, GroupNorm | 4 grupos, 64 canales | 64 × 16 × 23 | 128 |
| Bloque 3, pooling | ReLU; MaxPool 2 × 2, stride 2 | 64 × 8 × 11 | 0 |
| Agregación | AdaptiveAvgPool2d(1); Flatten | 64 | 0 |
| Regularización | Dropout, probabilidad 0,3 | 64 | 0 |
| Salida | Linear(64, 1), con bias | 1 logit | 65 |
| **Total** | | | **23 473** |

Las dimensiones impares se truncan según el pooling: 47 → 23 y 23 → 11. El promedio adaptativo permite consumir el mapa final sin una capa densa que dependa de todos sus píxeles. La salida binaria requiere un logit por ejemplo, no dos logits con softmax.

### 6.2. Convolución, normalización y agregación

Una convolución de entrada $H$ aplica filtros compartidos en posiciones de banda y tiempo:

$$
U[o,m,t]=\sum_c\sum_{a=0}^2\sum_{b=0}^2W[o,c,a,b]H[c,m+a-1,t+b-1].
$$

El padding completa los bordes. Un filtro combina cambios locales de energía en frecuencia y tiempo, pero no se ha inspeccionado su respuesta para atribuirle una función concreta de detección de sirena.

Para cada ejemplo y grupo de canales $g$, GroupNorm estima media y varianza sobre canales del grupo y posiciones espaciales, y aplica una transformación afín por canal:

$$
\widehat U_c=\gamma_c\frac{U_c-\mu_g}{\sqrt{\sigma_g^2+10^{-5}}}+\beta_c.
$$

Usa varianza con divisor poblacional y no guarda estadísticas corrientes dependientes del lote. No debe confundirse con la normalización de bandas ajustada previamente con train: son operaciones en etapas distintas. En inferencia, GroupNorm sigue calculando estadísticas del ejemplo. Esta distinción concuerda con [GroupNorm de PyTorch 2.11](https://docs.pytorch.org/docs/2.11/generated/torch.nn.GroupNorm.html).

ReLU aplica $\max(0,u)$ y MaxPool conserva el máximo local. Después de tres bloques, cada activación tiene un campo receptivo convolucional nominal de **22 × 22** posiciones de entrada y salto de 8, obtenido por la recurrencia $r'=r+(k-1)j$, $j'=js$. Es el campo de convolución/pooling: **GroupNorm usa todo el grupo espacial**, por lo que esa cifra no expresa una independencia estrictamente local de la salida. El pooling global también agrega toda la ventana.

Si $H\in\mathbb R^{64\times8\times11}$ es el mapa final:

$$
a_c=\frac1{88}\sum_{m=1}^8\sum_{t=1}^{11}H[c,m,t],\qquad
u=b+w^T\mathrm{Dropout}_{0,3}(a),\qquad p=\sigma(u).
$$

Durante entrenamiento, dropout anula coordenadas aleatorias y escala las restantes por $1/(1-0,3)$; en evaluación no anula ni reescala. La combinación de capas no lineales aprende sobre la matriz tiempo-frecuencia; no se reduce primero a medias de MFCC.

Inventario completo de los 11 tensores del `state_dict`: [inspeccion/state_tensors.csv](evidencia/04_cnn/inspeccion/state_tensors.csv). Los 23 473 elementos corresponden a parámetros; esta CNN no tiene buffers de BatchNorm. La arquitectura se define en la notebook y coincide con la cifra de [tables/latency.csv](evidencia/04_cnn/tables/latency.csv).

<a id="seccion-7"></a>

## 7. Entrenamiento profundo y parada interna

### 7.1. Tres roles distintos en cada fold

Cada fold externo separa sus propias filas de train con `StratifiedGroupKFold(5, shuffle=True, random_state=42+fold_externo)`. Se toma **solo la primera partición**: aproximadamente cuatro quintas partes reciben gradientes (`fit`) y una quinta parte controla parada (`early_stopping`). No se ejecuta una búsqueda interna de cinco folds completos. La validación del fold externo queda fuera de ambos subconjuntos y se utiliza para las métricas OOF.

| Fold externo | Rol | Segmentos | Positivos | Negativos | Grupos |
| --- | --- | --- | --- | --- | --- |
| 0 | early_stopping | 15 611 | 4 254 | 11 357 | 3 228 |
| 0 | fit | 63 503 | 17 968 | 45 535 | 13 036 |
| 0 | outer_validation | 39 200 | 11 052 | 28 148 | 8 099 |
| 1 | early_stopping | 15 672 | 4 777 | 10 895 | 3 228 |
| 1 | fit | 62 631 | 17 432 | 45 199 | 13 034 |
| 1 | outer_validation | 40 011 | 11 065 | 28 946 | 8 101 |
| 2 | early_stopping | 15 825 | 4 225 | 11 600 | 3 211 |
| 2 | fit | 63 386 | 17 892 | 45 494 | 12 989 |
| 2 | outer_validation | 39 103 | 11 157 | 27 946 | 8 163 |

Los grupos son disjuntos entre los tres roles. Las asignaciones internas de CNN y ResNet18 coinciden. Tanto la normalización de bandas como el peso positivo se calculan con `fit`, sin usar la parada interna. Los pesos positivos de estos ajustes son:

| Fold | Peso positivo $n_-/n_+$ |
| --- | --- |
| 0 | 2,534228 |
| 1 | 2,592875 |
| 2 | 2,542701 |

### 7.2. Función de pérdida y ponderación

Con un logit $u_i$, $p_i=\sigma(u_i)$ y $q=n_-/n_+$:

$$
\ell_i=-q\,y_i\log p_i-(1-y_i)\log(1-p_i),\qquad
\mathcal L=\frac1B\sum_{i=1}^B\ell_i.
$$

Se implementa con `BCEWithLogitsLoss(pos_weight=q)` para combinar sigmoide y entropía cruzada de forma numéricamente estable. Solo el término positivo se multiplica por $q$. El divisor es el número de ejemplos del lote, **no la suma de pesos**. En el ajuste final $q=85040/33274=2,555749$. La [API de PyTorch 2.11](https://docs.pytorch.org/docs/2.11/generated/torch.nn.BCEWithLogitsLoss.html) especifica esa semántica.

El gradiente por logit es $q(p_i-1)$ para un positivo y $p_i$ para un negativo, antes de dividir por $B$. El balance aumenta la contribución de positivos, pero no impone recall de 0,90 ni un FPR determinado. Como modifica el riesgo de ajuste, una salida sigmoide tampoco demuestra calibración con la prevalencia real.

### 7.3. AdamW, clipping y precisión mixta

AdamW usa momentos $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$ y $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$, con corrección de sesgo $\widehat m_t=m_t/(1-\beta_1^t)$ y $\widehat v_t=v_t/(1-\beta_2^t)$. La actualización, esquemáticamente, es:

$$
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon},\qquad
\beta_1=0,9,\ \beta_2=0,999,\ \epsilon=10^{-8}.
$$

La reducción por $\lambda$ está desacoplada del gradiente de la pérdida. En este código, el `weight_decay` se aplica a todos los parámetros incluidos en el optimizador, sin crear una excepción separada para biases o normalización. En ResNet, un parámetro congelado sin gradiente no recibe actualización. [Definición de AdamW](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html).

Antes de actualizar se desescala el gradiente de AMP y se limita su norma global a **5,0** mediante `clip_grad_norm_`:

$$
g\leftarrow g\min\left(1,\frac5{\|g\|_2+\epsilon_c}\right).
$$

El optimizador usa `zero_grad(set_to_none=True)`. AMP está habilitado en CUDA: `autocast` selecciona precisión por operación y `GradScaler` escala/desescala gradientes para reducir problemas de rango. Los pesos exportados permanecen en el dtype del estado guardado, principalmente `float32`; AMP no convierte todo el archivo a un modelo cuantizado.

`DataParallel` reparte el lote global de **64** en hasta dos GPU. La ejecución reporta dos **Tesla T4 de 15 360 MiB cada una**. Se fijan semillas de Python, NumPy y Torch; `cudnn.benchmark=False` y `cudnn.deterministic=True` reducen variación, pero no garantizan igualdad bit a bit entre versiones, dispositivos o rutas de precisión.

Las pérdidas y contadores de confusión de entrenamiento se acumulan en el dispositivo. Se transfieren resúmenes al final de la época y hay logs periódicos cada 100 batches. Se mantiene una comprobación de pérdida finita por lote; por tanto, no se ha eliminado toda sincronización. La evaluación usa modo `eval()` y contexto de inferencia, desactivando dropout y gradientes.

### 7.4. Regla exacta de parada

El presupuesto es **15 épocas máximas**, paciencia **3** y mejora mínima **0,0001**. La parada evalúa **BCE sin ponderación** sobre las filas internas, con probabilidades limitadas a $[10^{-7},1-10^{-7}]$ antes del logaritmo. Un checkpoint mejora cuando $\mathcal L_{stop}<\mathcal L_{mejor}-0,0001$. Cada época elegible sin esa mejora incrementa paciencia; al alcanzar tres, termina y se restaura el mejor estado.

Las métricas `inner_stop_f1`, recall y FPR usan umbral 0,5, pero **no deciden el checkpoint**: lo decide la pérdida. En ResNet, la elegibilidad comienza después de las dos épocas de cabeza. La validación fija no decide la parada.

`train_loss` es BCE ponderada; `train_loss_unweighted` es BCE sin ponderación calculada en batches de entrenamiento. `online_train_f1` agrega predicciones obtenidas mientras los parámetros cambian y dropout está activo. No es la evaluación retrospectiva del checkpoint final sobre todo train; una comparación de curvas debe tener presente esa diferencia.

### 7.5. Épocas efectivamente ejecutadas

| Ajuste | Épocas ejecutadas | Checkpoint elegido | BCE interna del elegido | F1 interno del elegido | Suma de tiempos de época, s |
| --- | --- | --- | --- | --- | --- |
| trial_0_fold_0 | 13 | 10 | 0,215357 | 0,898040 | 356,752425 |
| trial_0_fold_1 | 15 | 13 | 0,249963 | 0,886163 | 405,625700 |
| trial_0_fold_2 | 10 | 7 | 0,241204 | 0,886751 | 267,581057 |
| trial_1_fold_0 | 5 | 2 | 0,333893 | 0,827578 | 133,772275 |
| trial_1_fold_1 | 12 | 9 | 0,293656 | 0,843925 | 321,609648 |
| trial_1_fold_2 | 11 | 8 | 0,286434 | 0,863571 | 295,428149 |
| final_train | 10 | 10 | — | — | 423,136112 |

El ajuste final usa la mediana, redondeada, de las mejores épocas del candidato seleccionado: **10 épocas**. Se inicia un modelo nuevo, se estima normalización con todo train y se entrena con sus 118 314 segmentos durante ese número fijo. No se reutiliza directamente uno de los modelos de folds ni se crea un ensemble de sus checkpoints. En el ajuste final no existe subconjunto de parada y se conserva la última época prevista.

La suma de segundos de época no es idéntica a `fit_seconds`: el timer de ajuste incluye trabajo que rodea el registro de épocas, como escritura de logs y checkpoints. **No incluye calcular el normalizador, construir/inicializar el modelo ni preparar el optimizador y loaders**, porque empieza después de esas operaciones. El timer global `busqueda_cv` sí engloba esos preparativos de sus ajustes. `ajuste_final_train` reutiliza el `fit_seconds` devuelto y, por tanto, tampoco expresa todo el tiempo desde la llamada hasta tener el modelo final. Registro completo: [tables/epoch_history.csv](evidencia/04_cnn/tables/epoch_history.csv).

### 7.6. Evolución del ajuste final sobre train completo

| Época final | BCE ponderada online | BCE sin ponderar online | F1 macro online | Recall online | Tiempo de época, s |
| --- | --- | --- | --- | --- | --- |
| 1 | 0,601895 | 0,412849 | 0,788796 | 0,794073 | 41,257032 |
| 2 | 0,506241 | 0,345149 | 0,832289 | 0,827162 | 43,045795 |
| 3 | 0,465008 | 0,314627 | 0,849217 | 0,835728 | 42,124007 |
| 4 | 0,437013 | 0,294155 | 0,861517 | 0,845044 | 42,674850 |
| 5 | 0,417443 | 0,279960 | 0,868227 | 0,850123 | 42,958045 |
| 6 | 0,406296 | 0,272189 | 0,874200 | 0,853700 | 42,788343 |
| 7 | 0,395107 | 0,264700 | 0,878930 | 0,858358 | 41,882141 |
| 8 | 0,383506 | 0,256312 | 0,881524 | 0,861964 | 42,183206 |
| 9 | 0,380006 | 0,253668 | 0,884415 | 0,864158 | 42,014758 |
| 10 | 0,374158 | 0,249273 | 0,886223 | 0,863737 | 42,207936 |

Estas filas provienen de `final_train`, no de las curvas de parada de los folds. La pérdida ponderada es la que recibe gradientes; la otra permite comparar la escala de BCE sin balance, con las reservas sobre su medición online. No hay una columna de pérdida de validación fija por época: esa partición se conserva fuera del entrenamiento y se evalúa con el modelo final.

<a id="seccion-8"></a>

## 8. Búsqueda de candidatos y ajuste seleccionado

### 8.1. Presupuesto realmente explorado

| trial | lr | weight_decay | dropout |
| --- | --- | --- | --- |
| 0,000000 | 0,001000 | 0,000100 | 0,300000 |
| 1,000000 | 0,000300 | 0,001000 | 0,500000 |

Se compararon **2 candidatos × 3 folds = 6 ajustes de selección**. Los candidatos anteriores son el conjunto completo ejecutado; el informe no supone una búsqueda más amplia de kernels, arquitecturas o regularización.

### 8.2. Regla de elección y resultados de CV

La selección ordena por **F1 macro medio descendente**, después **recall medio descendente** y, solo si persiste empate, **suma del tiempo de ajuste ascendente**. Se exige completar los tres folds. Estas métricas utilizan el umbral predeterminado **0.5**, antes de buscar el umbral operativo OOF. La desviación estándar se calcula entre tres folds con divisor muestral (`ddof=1`); las barras de error no son intervalos de confianza.

| Candidato | F1 macro medio | Desv. estándar F1 | Recall medio | FPR medio | Tiempo de ajuste total, s | Folds completos |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0,878841 | 0,004698 | 0,793571 | 0,052284 | 1030,188064 | 3 |
| 1 | 0,846322 | 0,023548 | 0,742969 | 0,067832 | 751,010588 | 3 |

El candidato con lr=0,001, weight_decay=0,0001 y dropout=0,3 obtiene mayor F1 medio y menor dispersión observada que el candidato de lr=0,0003, weight_decay=0,001 y dropout=0,5. Como cambiaron tres parámetros simultáneamente, no puede atribuirse la mejora a uno de ellos por separado.

Detalle de los folds del candidato **0**:

| Fold | Filas con gradientes/ajuste | Filas de evaluación externa del fold | F1 macro | Recall | FPR | ROC-AUC | AP | Ajuste, s | Predicción, s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 63 503 | 39 200 | 0,878472 | 0,779225 | 0,045900 | 0,949115 | 0,912841 | 356,825583 | 10,448998 |
| 1 | 62 631 | 40 011 | 0,874339 | 0,824040 | 0,072307 | 0,946452 | 0,908703 | 405,715399 | 10,422145 |
| 2 | 63 386 | 39 103 | 0,883712 | 0,777449 | 0,038646 | 0,952216 | 0,919158 | 267,647082 | 10,117994 |

Para las redes, `train_segments` cuenta las filas `fit`, sin incluir la parada interna; por eso no coincide con todo el train del fold externo mostrado en la sección 5. Para los modelos clásicos coincide con ese train completo. Las filas de evaluación externa del fold siguen siendo las mismas entre modelos.

### 8.3. Configuración final congelada

| Parámetro congelado | Valor |
| --- | --- |
| lr | 0.001 |
| weight_decay | 0.0001 |
| dropout | 0.3 |
| final_epochs | 10 |

Se ajusta un estimador final nuevo con todas las filas de train. Los modelos clásicos vuelven a aprender sus transformaciones y parámetros; las redes vuelven a estimar su normalizador y entrenan el número de épocas fijado desde CV. El umbral se conserva a partir de las puntuaciones OOF del candidato elegido, sin buscar otro sobre la validación fija.

Fuentes: [tables/hyperparameter_candidates.csv](evidencia/04_cnn/tables/hyperparameter_candidates.csv), [tables/hyperparameter_summary.csv](evidencia/04_cnn/tables/hyperparameter_summary.csv), [tables/cv_results.csv](evidencia/04_cnn/tables/cv_results.csv) y [tables/selected_parameters.csv](evidencia/04_cnn/tables/selected_parameters.csv).

<a id="seccion-9"></a>

## 9. Umbral OOF y restricciones operativas

### 9.1. Barrido exacto de puntuaciones

El candidato elegido aporta una puntuación OOF a cada uno de los **118 314 segmentos de train**, predicha por un modelo cuyo conjunto de ajuste y parada no incluía su grupo. Se ordenan scores descendentes y se acumulan TP y FP por bloques de valores iguales. Los umbrales son los scores únicos; la comparación de clase es **`score >= threshold`**. Se añade un umbral `nextafter(max_score, +inf)` para incluir la solución sin predicciones positivas. Por eso el barrido contiene **12 773 umbrales candidatos**.

Tratar todos los empates como bloque evita resultados que dependan del orden arbitrario de ejemplos con el mismo score. El costo principal es ordenar, de orden $O(n\log n)$, y un barrido lineal; no se recalculan miles de matrices completas ni se utiliza una rejilla aproximada de 1 000 puntos.

El conjunto admisible se define:

$$
\mathcal T_{viable}=\{\tau:\mathrm{recall}(\tau)\ge0,90\ \land\ \mathrm{FPR}(\tau)<0,10\}.
$$

Si existe, se escoge el máximo F1 macro dentro de él. Si está vacío, se toma el máximo F1 macro de todo el barrido, con desempate por mayor recall y después mayor umbral. La igualdad FPR = 0,10 no sería suficiente: el límite es **estricto**.

### 9.2. Resultado y evidencia de inviabilidad

El reporte registra **`feasible_on_training_oof=false`**. La búsqueda no garantiza cumplimiento por el hecho de fijar estas metas. Los dos extremos siguientes permiten comprobar por qué no hay intersección:

| Punto del barrido OOF | Umbral | Recall | FPR | F1 macro |
| --- | --- | --- | --- | --- |
| Menor FPR que alcanza recall ≥ 0,90 | 0,232484341 | 0,900072128 | 0,160383349 | 0,836556235 |
| Mayor recall con FPR < 0,10 | 0,347647876 | 0,852978301 | 0,099929445 | 0,864350605 |

Para obtener el recall requerido en OOF se necesita al menos FPR **0,160383349**; al restringir FPR se alcanza como máximo recall **0,852978301**. Los valores describen el conjunto OOF de desarrollo y su curva discreta, no una frontera demostrada en producción.

El umbral final es **`0.5282900333404541`**, elegido por la alternativa explícita de máximo F1 macro. Comparación OOF recomputada:

| Política | Umbral | F1 macro | Precisión positiva | Recall | FPR | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Predeterminado | 0,500000 | 0,878785 | 0,855357 | 0,793532 | 0,052505 | 80 575 | 4 465 | 6 870 | 26 404 |
| Seleccionado OOF | 0,528290 | 0,879533 | 0,868355 | 0,783044 | 0,046449 | 81 090 | 3 950 | 7 219 | 26 055 |

La media de F1 entre folds de la sección anterior puede diferir del F1 calculado sobre todas las predicciones OOF juntas: F1 es una función no lineal de conteos, y los folds tienen tamaños diferentes. Esta tabla usa la unión de predicciones OOF, no el promedio de sus F1.

ROC-AUC y AP no cambian al mover el umbral sobre los mismos scores. El punto OOF seleccionado no es una nueva ejecución de entrenamiento ni la métrica de validación fija. No se recalibraron scores al pasar del modelo de folds al ajuste con train completo.

Fuentes: [reports/threshold_selection.json](evidencia/04_cnn/reports/threshold_selection.json) y [inspeccion/documentation_calculations.json](evidencia/04_cnn/inspeccion/documentation_calculations.json). El barrido íntegro permanece en `data/results/04_cnn_resultados/tables/threshold_search.csv`; las 118 314 predicciones originales, en `predictions/selected_oof_predictions.csv`.

<a id="seccion-10"></a>

## 10. Resultados de la validación interna fija

### 10.1. Unidad, umbral y fórmulas de evaluación

El estimador final entrenado con train se evalúa sobre **29 315 segmentos** de grupos reservados de los mismos corpus de desarrollo, usando el umbral OOF congelado. Los conteos se definen respecto de `EMERGENCY=1`: TP positivos detectados, FN positivos omitidos, FP negativos declarados emergencia y TN negativos correctamente rechazados.

$$
\mathrm{accuracy}=\frac{TP+TN}{TP+TN+FP+FN},\quad
\mathrm{precision}_+=\frac{TP}{TP+FP},\quad
\mathrm{recall}_+=\frac{TP}{TP+FN},\quad
\mathrm{FPR}=\frac{FP}{FP+TN}.
$$

$$
F1_+=\frac{2TP}{2TP+FP+FN},\qquad
F1_-=\frac{2TN}{2TN+FP+FN},\qquad
F1_{macro}=\frac{F1_++F1_-}2.
$$

Las clases se promedian con igual peso en F1 macro; esto no iguala el peso de corpus ni grupos. La precisión responde cuántas alertas son verdaderas dentro de la muestra; FPR responde qué fracción de los negativos genera alertas. No son complementos entre sí. Si una precisión o un F1 tiene denominador nulo, el evaluador usa `zero_division=0`.

ROC-AUC resume el ordenamiento continuo de positivos sobre negativos; admite la lectura $P(s_+>s_-)+\tfrac12P(s_+=s_-)$. AP acumula precisión a medida que crece recall por niveles de score:

$$
AP=\sum_k(R_k-R_{k-1})P_k,
$$

considerando puntos ordenados hacia recall creciente. AP no es necesariamente el área calculada por interpolación trapezoidal de la curva PR. Ambos valores usan scores, sin reducirlos primero a predicciones binarias. No se exportó una gráfica ROC en estas ejecuciones. [Convenciones de métricas de scikit-learn](https://scikit-learn.org/1.6/modules/model_evaluation.html#precision-recall-f-measure-metrics).

### 10.2. Matriz de confusión y métricas globales

| Etiqueta real / predicción | No emergencia | Emergencia |
| --- | --- | --- |
| No emergencia | 20 427 | 751 |
| Emergencia | 1 689 | 6 448 |

| Métrica por segmento | Valor |
| --- | --- |
| Accuracy | 0,916766 |
| Precisión de emergencia | 0,895680 |
| Recall de emergencia | 0,792430 |
| F1 macro | 0,892269 |
| FPR | 0,035461 |
| ROC-AUC | 0,956316 |
| Average Precision (AP) | 0,926079 |
| F1 de emergencia, derivado | 0,840897 |
| F1 de no emergencia, derivado | 0,943641 |
| Tasa de omisión, 1 − recall | 0,207570 |
| Especificidad, 1 − FPR | 0,964539 |

Ejemplos de sustitución: recall = 6448 / 8 137 = **0,792430**; FPR = 751 / 21 178 = **0,035461**; accuracy = (20427 + 6448) / 29 315 = **0,916766**. El clasificador que siempre declara no emergencia obtendría accuracy **0,722429**, por lo que accuracy sola no describe la detección de sirenas.

El modelo omite **1 689** ventanas positivas. El FPR global es inferior a 0,10, pero **recall < 0,90**, por lo que no satisface ambos criterios por segmento. Además, estos criterios están medidos en una partición interna de desarrollo; su cumplimiento externo no se ha comprobado.

### 10.3. Desglose por corpus

| Corpus | Segmentos | Precisión positiva | Recall | F1 macro | FPR | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A3Siren-Recordings | 352 | 0,010695 | 0,050000 | 0,275058 | 0,592949 | 0,065665 | 0,061908 |
| AudioSet-EV v2 | 28 630 | 0,917048 | 0,791667 | 0,899115 | 0,027043 | 0,958671 | 0,930096 |
| sireNNet | 333 | 0,983122 | 0,935743 | 0,923868 | 0,047619 | 0,975760 | 0,993136 |

| Corpus | TN | FP | FN | TP |
| --- | --- | --- | --- | --- |
| A3Siren-Recordings | 127 | 185 | 38 | 2 |
| AudioSet-EV v2 | 20 220 | 562 | 1 635 | 6 213 |
| sireNNet | 80 | 4 | 16 | 233 |

La CNN obtiene F1 macro cercano a 0,899 en AudioSet y 0,924 en sireNNet, con FPR menores que 0,05 en ambos. En A3 detecta dos de los 40 positivos y genera 185 FP entre 312 negativos. Su AP de aproximadamente 0,062 y ROC-AUC de 0,066 describen una transferencia deficiente en estos tracks. La buena cifra global no caracteriza de forma uniforme los corpus.

AudioSet aporta **28 630 de 29 315 segmentos (97,663 %)**. Las métricas globales están dominadas por ese corpus, aunque F1 macro balancee las dos clases. A3 aporta 352 segmentos, pero provienen de **solo dos tracks**: uno de emergencia con 40 ventanas y otro negativo con 312. Una gran cantidad de errores puede estar concentrada en una misma grabación; no corresponde presentarla como 352 experimentos independientes.

La prevalencia positiva es 27,4118 % en AudioSet, 74,7748 % en sireNNet y 11,3636 % en A3. Una política siempre negativa lograría accuracy de 88,6364 % en A3: por eso su accuracy se interpreta junto con recall, FPR, ranking y conteos. Diferencias de captura, etiqueta y distribución pueden explicar parte de la variación, pero estos resultados no identifican causalmente cuál factor produjo el fallo. No se invierten etiquetas ni se retoca el umbral mirando A3 después de evaluar.

### 10.4. Intervalos por bootstrap de grupos

La notebook genera **500 repeticiones** con semilla 42. En cada repetición toma 6 090 grupos de validación con reemplazo y concatena todos los segmentos de cada grupo elegido, repitiéndolos cuando el grupo se selecciona varias veces. Después calcula las métricas por segmento con el modelo y umbral fijos. Los percentiles 2,5 y 97,5 forman los intervalos aproximados del 95 %:

| Métrica | Límite inferior | Límite superior | Repeticiones finitas |
| --- | --- | --- | --- |
| f1_macro | 0,874295 | 0,904462 | 500 |
| recall_emergency | 0,775118 | 0,808688 | 500 |
| fpr | 0,024489 | 0,053963 | 500 |
| roc_auc | 0,948339 | 0,962359 | 500 |

El remuestreo por grupos conserva el bloque de ventanas correlacionadas y los canales agrupados. Las métricas de cada remuestreo continúan siendo por segmento: un grupo largo aporta más filas. El bootstrap no está estratificado por corpus ni etiqueta y no vuelve a entrenar ni seleccionar hiperparámetros o umbral. Por tanto, los intervalos son condicionales al modelo y partición observados, no incluyen toda la incertidumbre del proceso de modelado y no son una validación externa. Con solo dos tracks A3, tampoco producen evidencia abundante e independiente de ese dominio.

El límite superior de recall, **0,808688**, sigue por debajo de 0,90. El punto de FPR y su intervalo se interpretan por separado; que el punto esté debajo de 0,10 no hace que todo el intervalo tenga que estarlo. La comparación de intervalos entre modelos no sustituye una prueba pareada sobre sus errores compartidos.

### 10.5. Agregación por grupo: «algún segmento positivo»

Para cada par `(dataset, group_id)` se define:

$$
y_g=\max_{i\in g}y_i,\qquad s_g=\max_{i\in g}s_i,\qquad
\widehat y_g=\mathbf1[s_g\ge\tau].
$$

Hay **6 090 grupos**, con 1 828 positivos y 4 262 negativos. Con el mismo umbral, la matriz es:

| Etiqueta real del grupo / predicción | No emergencia | Emergencia |
| --- | --- | --- |
| No emergencia | 3 983 | 279 |
| Emergencia | 102 | 1 726 |

| Métrica por grupo | Valor |
| --- | --- |
| Accuracy | 0,937438 |
| Precisión de emergencia | 0,860848 |
| Recall de emergencia | 0,944201 |
| F1 macro | 0,927477 |
| FPR | 0,065462 |
| ROC-AUC | 0,984935 |
| Average Precision (AP) | 0,971464 |

El máximo puede detectar un grupo positivo aunque muchas de sus ventanas se omitan, y puede convertir un grupo negativo en falsa alarma con un único segmento erróneo. Bajo una hipótesis ilustrativa de ventanas independientes con FPR $p$, la probabilidad de alguna falsa alarma en $m$ ventanas sería $1-(1-p)^m$; **esa independencia no se cumple necesariamente aquí** y la expresión no se utiliza para calcular el reporte.

El resultado por grupo no reemplaza al resultado por segmento ni demuestra sensibilidad por evento temporal, latencia de detección o falsas alarmas por hora. Puede superar una meta en una unidad de análisis mientras falla en otra. A3 agrupa canales del track; AudioSet y sireNNet utilizan las dependencias recuperadas. Definición exportada: [reports/group_metric_definition.json](evidencia/04_cnn/reports/group_metric_definition.json).

Fuentes numéricas: [tables/validation_metrics.csv](evidencia/04_cnn/tables/validation_metrics.csv), [tables/validation_by_corpus.csv](evidencia/04_cnn/tables/validation_by_corpus.csv), [tables/validation_cluster_bootstrap_ci.csv](evidencia/04_cnn/tables/validation_cluster_bootstrap_ci.csv) y [tables/validation_group_any_event.csv](evidencia/04_cnn/tables/validation_group_any_event.csv). Las métricas globales, por corpus y por grupo se recomputaron desde `validation_predictions.csv` y coinciden con estas exportaciones.

<a id="seccion-11"></a>

## 11. Figuras originales y lectura técnica

Todas las imágenes de esta sección provienen de la ejecución entregada. Se conservaron sin redibujarlas ni agregar puntos o anotaciones a sus archivos. Cada PNG tiene una versión SVG para ampliar o exportar.

### 11.1. F1 macro de candidatos

![F1 macro de candidatos de CNN](figuras/04_cnn/validacion_cruzada.png)

Cada barra es el F1 macro medio de tres folds al umbral predeterminado; las barras de error muestran desviación estándar muestral. Gana el candidato 0, con media 0,878841. El eje de candidato es un identificador de configuración, no el número de épocas ni una progresión temporal. El gráfico no muestra el F1 posterior al ajuste del umbral.

[Versión SVG](figuras/04_cnn/validacion_cruzada.svg).

### 11.2. Matriz de confusión de validación fija

![Matriz de confusión de validación fija de CNN](figuras/04_cnn/validacion_matriz_confusion.png)

Filas: etiqueta real; columnas: predicción. Los conteos son TN=20427, FP=751, FN=1689, TP=6448, usando el umbral 0.5282900333404541. Son cantidades absolutas por segmento, sin normalización por corpus ni duración. El color resalta volumen, por lo que debe leerse junto a los conteos y tasas.

[Versión SVG](figuras/04_cnn/validacion_matriz_confusion.svg).

### 11.3. Curva precisión–recall

![Curva precisión–recall de CNN](figuras/04_cnn/validacion_precision_recall.png)

La curva barre los scores del modelo final sobre la validación fija; AP=0,926079. Su extremo de recall completo refleja la prevalencia positiva global, aproximadamente 0,277571. El punto operativo congelado tiene recall 0,792430 y precisión 0,895680; la imagen original no lo marca por separado. La curva PR no representa FPR ni demuestra por sí sola el cumplimiento de recall/FPR simultáneos.

[Versión SVG](figuras/04_cnn/validacion_precision_recall.svg).

### 11.4. Costos registrados

![Costos registrados de CNN](figuras/04_cnn/costo_computacional.png)

La figura exportada grafica las fases de `runtime_rows`: búsqueda CV, ajuste final y predicción de validación. La tabla completa de costos añade descubrimiento, verificación y lectura. Los ejes están en segundos; no incluyen toda la experiencia de ejecución, esperas manuales o descarga como fases independientes. En redes, hay timers anidados de verificación y lectura que no deben sumarse como tiempos disjuntos.

[Versión SVG](figuras/04_cnn/costo_computacional.svg).

### 11.5. BCE del subconjunto interno de parada

![BCE del subconjunto interno de parada de CNN](figuras/04_cnn/curvas_entrenamiento.png)

Las pérdidas internas mínimas del candidato 0 están en épocas 10, 13 y 7; el fold 1 alcanza el presupuesto de 15 épocas sin consumir toda la paciencia después de su mejor época. Las curvas muestran oscilaciones de parada mientras la pérdida online de train desciende; esto justifica restaurar el checkpoint elegido por BCE en lugar de asumir que la última época es la mejor. Las líneas son los tres folds del candidato seleccionado, no tres corpus y no la validación fija.

[Versión SVG](figuras/04_cnn/curvas_entrenamiento.svg).

### 11.6. Pérdidas comparables y F1 de parada

![Pérdidas comparables y F1 de parada de CNN](figuras/04_cnn/loss_f1_detalle.png)

Panel izquierdo: BCE sin ponderar de entrenamiento online en líneas discontinuas y BCE de parada en líneas continuas. Panel derecho: F1 macro de parada con umbral 0,5. Ambos paneles usan el candidato elegido; el modelo final no se vuelve a escoger mirando estas figuras. La BCE de parada decide el checkpoint, aunque el máximo de F1 ocurra en otra época. La pérdida online se midió con pesos cambiantes y dropout activo.

[Versión SVG](figuras/04_cnn/loss_f1_detalle.svg).

<a id="seccion-12"></a>

## 12. Entorno, tiempos y costo computacional

### 12.1. Entorno efectivamente registrado

| Componente | Valor |
| --- | --- |
| Python exportado | 3.13.15 |
| NumPy | 2.1.3 |
| pandas | 2.3.3 |
| scikit-learn | 1.6.1 |
| SciPy | 1.16.3 |
| Matplotlib | 3.10.0 |
| joblib | 1.6.0 |
| CPU disponibles | 4 |
| RAM física del entorno, GiB | 31,347 |
| PyTorch | 2.11.0+cu128 |
| Torchvision | 0.26.0+cu128 |
| CUDA | 12.8 |
| cuDNN, código reportado | 91 900 |
| GPU utilizadas | 2 |
| GPU | 2 × Tesla T4, 15 360 MiB cada una |
| Paralelismo | DataParallel |
| AMP | Habilitado |
| Lote global | 64 |
| Workers del DataLoader | 0 |

Estas son las versiones del **entrenamiento en Kaggle**, no las del entorno local usado después para leer tablas. El entorno de preparación registró Librosa **0.11.0**, SoundFile **0.14.0** y remuestreo `soxr_hq`. Las rutas originales de datasets, reserva de disco, `prepared_cache`, heartbeat y demás opciones están conservadas en [config/training_config.json](evidencia/04_cnn/config/training_config.json) y [la configuración de preparación](evidencia/preparacion/config/preparation_config.json).

### 12.2. Tiempos exportados de las fases

| Fase | Segundos |
| --- | --- |
| descubrimiento_insumos | 31,124895 |
| verificacion_hashes_publicados | 2,712350 |
| lectura_y_auditoria_metadata | 8,602347 |
| verificacion_hashes_publicados | 28,992140 |
| lectura_representaciones | 29,878598 |
| busqueda_cv | 1859,606381 |
| ajuste_final_train | 423,217108 |
| prediccion_validacion | 7,947017 |

Los tiempos son mediciones de esa ejecución, con los insumos disponibles y el estado de caché correspondiente. No representan estimaciones de complejidad teórica ni un benchmark repetido con igualdad de condiciones entre todas las ejecuciones. En redes, la segunda verificación de hashes está dentro del timer de lectura de representaciones: sumar ambas filas contaría trabajo repetido en el total. El tiempo de búsqueda incluye el trabajo de sus candidatos y folds; tampoco debe sumarse otra vez el total de `fit_seconds` de esos folds.

Para referencia del trabajo compartido anterior, la preparación registró descubrimiento 73,617 s, inventario 500,400 s y extracción 2 016,066 s. Su timer `total_through_validation` es 5 126,702 s y **no es otra fase disjunta para sumar**. Ese costo común de preparar audio no está incluido en la latencia de ningún clasificador.

### 12.3. Inferencia sobre representaciones precomputadas

| Medición | Valor |
| --- | --- |
| p50, lote de un ejemplo en una GPU, ms | 0,966629 |
| p95, lote de un ejemplo en una GPU, ms | 1,614517 |
| Repeticiones de latencia | 100 |
| Lote global de throughput | 64 |
| GPU de throughput | 2 |
| Throughput, segmentos/s | 5159,329569 |
| Parámetros | 23 473 |
| Archivo final, MiB | 0,096283 |

La latencia usa el modelo sin wrapper, un tensor de un ejemplo ya normalizado y residente en una GPU. Incluye solo el forward con AMP, tras 20 calentamientos y con sincronizaciones CUDA antes/después de cada una de 100 repeticiones. El throughput usa el modelo final con DataParallel, tres forwards de calentamiento y diez forwards cronometrados con el lote global indicado. No incluye disco, traslado CPU→GPU, decodificación, STFT, Log-Mel, normalización ni segmentación. No es latencia de una alerta extremo a extremo.

Los eventos registran RSS altos al cierre: aproximadamente 24,829 GiB. Esa lectura puntual no es un pico independiente de memoria GPU ni se atribuye únicamente a parámetros: incluye estructuras, temporales y páginas de representaciones mapeadas. `epoch_history.csv` registra por GPU máximos asignados y reservados; memoria reservada por el allocator no equivale a memoria total física ocupada por todo el proceso.

Los modelos clásicos se midieron en CPU y las redes en GPU. Las tasas no son una comparación de rendimiento a igualdad de hardware, y una notebook en Kaggle no prueba las restricciones temporales de un prototipo en vehículo. Ventana de 3 s, salto de 1,5 s y tiempos de preparación son componentes adicionales para una evaluación de respuesta completa.

Fuentes: [reports/environment.json](evidencia/04_cnn/reports/environment.json), [tables/computational_cost.csv](evidencia/04_cnn/tables/computational_cost.csv) y [tables/latency.csv](evidencia/04_cnn/tables/latency.csv).

<a id="seccion-13"></a>

## 13. Artefacto final, trazabilidad y recarga

### 13.1. Qué se conserva

| Archivo o grupo | Contenido y función |
| --- | --- |
| `data/models/04_cnn_modelos/models/model.pt` | Estado final del modelo ajustado con todas las filas de train |
| [models/inference.json](evidencia/04_cnn/models/inference.json) | Umbral y contrato de inferencia; también normalización en redes |
| [config/frozen_experiment.json](evidencia/04_cnn/config/frozen_experiment.json) | Parámetros elegidos, umbral, hashes de entradas y folds, y alcance de desarrollo |
| [config/training_config.json](evidencia/04_cnn/config/training_config.json) | Configuración efectiva completa de la ejecución |
| [reports/artifact_hashes.json](evidencia/04_cnn/reports/artifact_hashes.json) | Hashes publicados de configuración, insumos y artefactos |
| `predictions/validation_predictions.csv` en resultados originales | segment_id, corpus, archivo, grupo, etiqueta, score y clase de validación fija |
| `predictions/selected_oof_predictions.csv` en resultados originales | Predicción OOF elegida para cada fila de train |
| `logs/events.jsonl` y `logs/training.log` en resultados originales | Eventos, etapas, duración y diagnósticos de recursos |

El modelo entregado ocupa **100 960 bytes**. Su SHA-256 se recalculó localmente y coincide con el hash publicado:

```text
78734a55aefefc3c71f9be1b0e5c9a05375734f7e54870ab7d2febca24bb3632
```

El `inference.json` y todas las configuraciones de `data/models` también coinciden con sus equivalentes y hashes de `data/results`. Un peso sin su transformación de entrada, normalización, orden de características y umbral no constituye el mismo predictor efectivo.

### 13.2. Identidad del paquete y asignaciones

El archivo [PREPARATION_READY.json](evidencia/preparacion/PREPARATION_READY.json) certifica la preparación validada y registra:

```text
schema_version: doppler-prepared-v3
config_sha256: 99ce56b293aa68b45200ecd833b58da177d6391d4834d79cc8d7328b3a0cfba3
file_split_sha256: 59b34975f30f52ad8725ae116957c2079501d2de9321a68c72999bef363b19f7
files: 32204
segments: 153993
features_scaled: false
external_evaluated: false
```

`config_sha256` corresponde a la identidad de configuración serializada canónicamente por la preparación. El hash del **archivo** `config/preparation_config.json` en `input_sha256` es `daed1f14b16c9cf9653abc2095d66592e125f582e69e1b3497691368f8763521`: son objetos de hash distintos, por lo que esos valores no deben compararse como si fuesen el mismo digest.

Las asignaciones OOF tienen hash `3528557381804ea80cf8a9fd230bbe0c37f24e721094162609a2c41f83a2154d`. La reanudación de preparación contrasta fuentes, configuración, plan de filas y checksums antes de reutilizar resultados; los ZIP se generan temporalmente y se verifican antes de publicar el nombre final. Esas garantías corresponden al código de preparación, no a que el entrenamiento final guarde todo el estado para reanudar un optimizador.

**Estado del snapshot de preparación:** su `reports/run_status.json` contiene `RUNNING`, mientras `PREPARATION_READY.json` indica `READY` y las validaciones finales pasan. El código crea el ZIP completo antes de escribir `preparation_complete` y `COMPLETE`; por eso un snapshot de ese ZIP conserva el estado anterior. No se ha reescrito la evidencia. Los cinco reportes de entrenamiento sí registran `COMPLETE`. El READY habilita el consumo del paquete validado; no debe citarse ese JSON de estado de preparación como si dijera COMPLETE.

### 13.3. Contenido del checkpoint y prueba registrada

`model.pt` contiene `state_dict`, 64 medias y 64 desviaciones, `model_kind`, parámetros seleccionados, umbral, tipo de score, `input_shape=[64,94]`, `class_map` y configuración. El estado se guarda sin prefijo de wrapper multi GPU. La función `load_exported_model` usa `torch.load(..., weights_only=True)`, construye la arquitectura sin inicialización preentrenada, carga con `strict=True` y devuelve modelo en `eval()`, normalizador y umbral.

La ejecución registra una recarga **aprobada sobre 8 ejemplos de validación**, con `rtol=0.0001` y `atol=1e-05`. El criterio numérico es $|a-b|\le\mathrm{atol}+\mathrm{rtol}|b|$, junto con igualdad del umbral. En CNN se registra una comparación de recarga bajo la ruta de inferencia definida en su notebook. No se publicó una certificación de igualdad entre todos los tipos de GPU o dtypes posibles.

La prueba detecta diferencias en ese subconjunto, pero no es una reevaluación completa del modelo recargado sobre 29 315 segmentos. El SHA-256 verifica integridad del archivo completo, mientras la prueba de ocho scores verifica comportamiento de una muestra. Reporte original: [reports/model_reload_check.json](evidencia/04_cnn/reports/model_reload_check.json).

### 13.4. Uso correcto de la exportación

Una vez disponibles las clases y la función de recarga definidas en la notebook, el flujo de inferencia es cargar el artefacto, aplicar el mismo extractor de Log-Mel a una ventana de tres segundos y normalizar con los parámetros guardados. El siguiente ejemplo parte de una matriz **ya preparada**, no de audio crudo:

```python
from pathlib import Path
import numpy as np
import torch

artifact_path = Path("data/models/04_cnn_modelos/models/model.pt")
model, (mean, std), threshold = load_exported_model(artifact_path, device="cpu")
# prepared_logmel: matriz float32 (64, 94), con el contrato de preparación v3.
normalized = (prepared_logmel - mean[:, None]) / std[:, None]
batch = torch.from_numpy(normalized.astype(np.float32))[None, None]
with torch.inference_mode():
    score = torch.sigmoid(model(batch)).item()
prediction = int(score >= threshold)
```

La entrada es un canal en ambos modelos; ResNet repite los canales dentro de `forward`. El snippet CPU sirve para explicar la transformación y no reproduce el benchmark AMP/T4. No debe compararse como si fuese la misma ruta numérica del reporte. Las etiquetas de producción no se usan para inferir y el umbral no se vuelve a estimar por batch.

La exportación final **no contiene estado de AdamW, GradScaler ni todos los estados aleatorios de entrenamiento**. Los checkpoints de CV conservan pesos, normalización, parámetros y época elegida, pero no constituyen un checkpoint completo para continuar exactamente el optimizador. Los eventos TensorBoard originales complementan el CSV; no fueron copiados como resultados nuevos.

### 13.5. Reproducción del experimento

1. Ejecutar o montar la misma preparación v3 con las fuentes declaradas, tasas, ventanas y reglas de etiquetas registradas. Contrastar READY, hashes de insumos y contratos; no sustituir otra preparación solo porque sus columnas se llamen igual.
2. Adjuntar los insumos a Kaggle y ejecutar la notebook autocontenida de este modelo con la configuración efectiva entregada. Para redes se requieren GPU; para la inicialización de ResNet se necesitan los pesos oficiales descargados o un archivo configurado.
3. Mantener el split fijo, grupos, semilla, tres folds y conjunto de candidatos. Conservar la asignación interna de parada para las redes y ajustar transformaciones solo con filas de ajuste.
4. Elegir candidato con la regla de F1 medio, obtener OOF, fijar umbral con restricciones y fallback, y ajustar nuevamente con train completo.
5. Evaluar validación fija una vez con la política congelada y comparar conteos, métricas, artefactos y entorno. Si cambian insumos o configuración, etiquetar la salida como otra ejecución; no presentar variación como reproducción exacta.
6. Mantener LSSiren fuera de selección, escalamiento, gradientes, parada y umbral. Su evaluación pertenece a la fase externa pendiente del proyecto.

La semilla favorece repetibilidad, pero no garantiza identidad binaria de archivos o scores en distinto software/hardware. El repositorio, las exportaciones y sus hashes permiten rastrear esta ejecución; no se publicó un commit de Git ni un lockfile completo del entorno dentro del experimento, así que no se añade un identificador ficticio.

<a id="seccion-14"></a>

## 14. Alcance de las conclusiones y trabajo pendiente

- La red aprovecha evolución tiempo-frecuencia con solo 23 473 parámetros, pero los resultados no identifican qué patrón acústico aprendió cada filtro.
- El F1 global de 0,892 y el FPR bajo conviven con recall de 0,792 y dos TP de 40 positivos A3; una mejora global no resuelve la cobertura entre corpus.
- El training online, la parada interna y los folds externos son mediciones diferentes. Las curvas de parada no son resultados de generalización externa.

Las conclusiones están limitadas por las etiquetas heredadas, dependencias de aumentos de sireNNet no identificadas, correlación de ventanas, contribución de grupos largos y solo dos tracks A3 en validación fija. El entrenamiento balancea clases, pero no equilibra automáticamente corpus ni grupos. Comparar los cinco modelos sobre los mismos datos es útil para el desarrollo; una elección basada en esa comparación debe seguir tratándose como selección interna.

Permanecen pendientes, conforme al alcance definido: **evaluación externa sobre LSSiren**, evaluación de robustez y prototipo. No hay resultados nuevos de ruido, SNR, distancia, localización, falsos positivos por hora, inferencia en un vehículo o latencia de detección temporal en estos archivos. Las tasas del benchmark se refieren a representaciones precomputadas y al hardware de Kaggle.

El fallo de A3 merece examinar anotaciones y condiciones acústicas dentro de las fases previstas, conservando la trazabilidad y sin reajustar silenciosamente este experimento. Las diferencias de corpus son una observación; atribuirlas a un micrófono, una frecuencia o una estrategia de entrenamiento concretos sería una hipótesis que aún requiere contraste.

## Fuentes del proyecto y navegación

- [Criterios de éxito: recall ≥ 0,90 y FPR < 0,10](../2/2.4/criterios-de-exito.md).
- [Protocolo de evaluación interna](../4/4.5/4.5.5/protocolo-de-evaluacion-interna.md).
- [Protocolo de generalización entre corpus](../4/4.5/4.5.6/protocolo-de-generalizacion-entre-corpus.md).
- [Contrato de preparación y convenciones](../contrato-preparacion.md).
- [Índice comparativo de los cinco modelos](README.md).
- [Procedencia, validación y alcance de las copias documentales](evidencia/README.md).

Las referencias de bibliotecas junto a cada fórmula aclaran convenciones de las API; los conteos, parámetros, tiempos y resultados de este informe proceden de las notebooks y exportaciones entregadas. No se reemplazó ninguna ejecución histórica ni se generaron nuevas puntuaciones de modelos para escribirlo.
