# 3.1.2. Composición de los datasets

La composición de las cuatro fuentes presenta diferencias importantes en escala y estructura. AudioSet-EV v2 constituye el corpus de mayor tamaño disponible en el proyecto, con 28 816 archivos físicos; LSSiren contiene 1 834; sireNNet, 1 675; y A3Siren-Recordings, 104 archivos correspondientes a 13 tomas multicanal. En términos globales, la auditoría identificó 32 429 archivos WAV, de los cuales 32 427 fueron utilizables para las etapas posteriores.

La estructura de las etiquetas tampoco es uniforme. AudioSet-EV v2 mantiene una organización Positive/Negative y metadata procedente de la ontología de AudioSet; LSSiren presenta Ambulance/Road; sireNNet utiliza cuatro categorías ambulance, firetruck, police y traffic; mientras A3Siren-Recordings presenta Siren/Noise y una organización adicional basada en tracks y canales.

| **Dataset**        | **Archivos físicos** | **Archivos válidos** | **Grupos observados**                                         | **Frecuencia de muestreo**     | **Canales**                            | **Duración mediana (s)** | **Unidad de análisis** |
|--------------------|----------------------|----------------------|---------------------------------------------------------------|--------------------------------|----------------------------------------|--------------------------|------------------------|
| AudioSet-EV v2     | 28816                | 28814                | Negative_files (20 916); Positive_files (7 900)               | 32 kHz                         | Mono                                   | 10.00                    | Archivo                |
| LSSiren            | 1834                 | 1834                 | Ambulance (932); Road (902)                                   | 8 / 22.05 / 44.1 / 48 / 96 kHz | Mono / estéreo                         | 8.73                     | Archivo                |
| sireNNet           | 1675                 | 1675                 | ambulance (400); firetruck (400); police (454); traffic (421) | 44.1 kHz                       | Estéreo                                | 3.00                     | Archivo                |
| A3Siren-Recordings | 104                  | 104                  | Siren (56 WAV; 7 tracks); Noise (48 WAV; 6 tracks)            | 44.1 kHz                       | 8 canales lógicos (WAV mono por canal) | 53.00 por track          | Track                  |

<span id="_Toc242396247" class="anchor"></span>Tabla 7. Resumen de los datasets
