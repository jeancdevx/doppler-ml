# 4.4. Ingeniería de Características

La ingeniería de características tuvo como propósito transformar cada segmento acústico preparado en una representación numérica capaz de describir diferentes dimensiones de la señal sonora. A diferencia de la etapa anterior, centrada en armonizar y transformar temporalmente las grabaciones, en esta fase se construyeron descriptores que resumen propiedades energéticas, temporales, espectrales y cepstrales de cada segmento.

El diseño se fundamentó en los patrones observados durante el análisis exploratorio, donde se identificó variación medible en RMS, ZCR, frecuencia dominante, centroide espectral, ancho de banda, roll-off, flujo espectral, flatness y los 13 MFCC. Estos descriptores fueron considerados candidatos para un análisis posterior, sin establecer anticipadamente que alguno de ellos constituiría por sí solo una regla de clasificación.

La ingeniería de características se organizó en dos representaciones complementarias. La primera corresponde a una representación tabular de 21 variables acústicas por segmento, destinada a modelos que reciben vectores numéricos. La segunda corresponde a la representación Log-Mel, conservada como matriz tiempo-frecuencia para los experimentos posteriores con modelos de espectrograma.
