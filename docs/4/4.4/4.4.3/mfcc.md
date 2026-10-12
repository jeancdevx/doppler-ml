# 4.4.3. MFCC

La tercera familia correspondió a los Mel-Frequency Cepstral Coefficients (MFCC). En el pipeline se calcularon 13 coeficientes por segmento, utilizando la representación Mel con 64 bandas y los mismos parámetros espectrales definidos para el resto de la preparación.

Los MFCC proporcionan una descripción compacta de la estructura espectral de la señal después de proyectar su contenido frecuencial sobre una escala Mel. En este caso, cada segmento quedó representado por:

$$MFCC = \left\{ mfcc_{1},mfcc_{2},\ldots,mfcc_{13} \right\}$$

Cada coeficiente fue posteriormente resumido mediante su valor medio a lo largo de las tramas del segmento, de modo que los 13 coeficientes contribuyeron como variables escalares al vector tabular final.

La inclusión de los MFCC se fundamenta también en el análisis exploratorio realizado sobre sireNNet, donde los perfiles de los 13 coeficientes presentaron variaciones entre las categorías observadas, aunque acompañadas de variabilidad intraclase. Por esta razón, los MFCC se incorporaron como representación complementaria de la estructura acústica y no como un conjunto previamente seleccionado por rendimiento predictivo.
