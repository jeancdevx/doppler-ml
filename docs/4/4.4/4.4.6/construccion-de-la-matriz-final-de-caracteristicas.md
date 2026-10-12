# 4.4.6. Construcción de la matriz final de características

A partir de los descriptores extraídos se construyó una representación tabular de 21 variables por segmento, compuesta por:

$$8\ \text{descriptores~acústicos} + 13\ \text{MFCC} = 21\ \text{características}$$

Cada segmento queda representado por el vector x = \[RMS, ZCR, frecuencia dominante, centroide espectral, ancho de banda espectral, roll-off, flujo espectral, planitud espectral, MFCC1, …, MFCC13\], con un total de 21 variables. Los descriptores calculados por tramas se resumen conforme a la operación registrada en el extractor; en los MFCC se utiliza la media temporal reportada.

Esta estructura permite que una fila no constituya únicamente un vector numérico aislado, sino una observación completamente trazable dentro del pipeline experimental.

| **Representación**        | **Contenido**                                               | **Dimensión**                           | **Unidad**               | **Uso**                                  |
|---------------------------|-------------------------------------------------------------|-----------------------------------------|--------------------------|------------------------------------------|
| Características tabulares | 8 descriptores acústicos y las medias temporales de 13 MFCC | 21 valores por segmento                 | Ventana de tres segundos | Regresión logística, Random Forest y SVM |
| Log-Mel                   | 64 bandas Mel entre 20 y 8 000 Hz                           | 64 × T; T debe verificarse en la salida | Ventana de tres segundos | CNN compacta y ResNet18                  |

<span id="_Toc242396257" class="anchor"></span>Tabla 17. Representaciones y características generadas
