# 3.5.6. Síntesis de los hallazgos iniciales

Los cinco hallazgos muestran que la principal característica de los datos analizados es su heterogeneidad multidimensional. Las diferencias no se limitan al número de observaciones, sino que abarcan la estructura de las señales, su duración, las condiciones técnicas de captura y el comportamiento de las variables acústicas.

En términos temporales y estructurales, los corpus presentan diferentes unidades y escalas de duración. En términos técnicos, existen diferencias de frecuencia de muestreo y configuración de canales. Desde el punto de vista acústico, variables como RMS, ZCR y los descriptores espectrales presentan diferencias observables entre determinados grupos, pero también distintos grados de solapamiento. Finalmente, A3Siren evidencia que incluso dentro de una misma toma la posición o canal de captura puede modificar las características medidas.

Por tanto, los resultados del EDA no justifican reducir el problema a una única característica acústica ni asumir que todos los corpus pueden procesarse mediante un mismo procedimiento sin adaptación. En su lugar, proporcionan evidencia para diseñar una etapa de preparación que considere explícitamente la heterogeneidad temporal, técnica y acústica observada.

| **ID** | **Hallazgo**                           | **Evidencia sintetizada**                                                                             | **Implicación para el Capítulo 4**                                              |
|--------|----------------------------------------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| H1     | Heterogeneidad estructural             | Difieren la tasa de muestreo, el número de canales y la duración entre corpus.                        | Evaluar armonización de tasa, canales y duración en la preparación.             |
| H2     | Escalas temporales distintas           | Medianas: 10.00 s (AudioSet-EV), 8.71 s (LSSiren), 3.00 s (sireNNet) y 53.00 s por track (A3Siren).   | Evaluar segmentación o ventaneo sin asumir una duración única.                  |
| H3     | Diferenciación acústica parcial        | RMS y centroide muestran diferencias descriptivas con distintos grados de solapamiento entre grupos.  | Evaluar una representación multivariable; no usar una sola variable como regla. |
| H4     | Variabilidad intraclase y entre corpus | El centroide presenta dispersión dentro de los corpus y diferencias entre sus medianas.               | Conservar la identidad del corpus al comparar o combinar representaciones.      |
| H5     | Dependencia de condiciones de captura  | LSSiren mezcla cinco tasas y mono/estéreo; A3Siren distribuye cada toma en ocho canales dependientes. | Evaluar tratamiento de canales/SR y particionar A3Siren por track.              |

<span id="_Toc242396253" class="anchor"></span>Tabla 13. Hallazgos iniciales y sus implicaciones para la preparación.
