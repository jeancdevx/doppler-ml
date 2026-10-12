# 4.3.4. Transformación Log-Mel

A partir de la STFT se construyó una representación basada en 64 bandas Mel, utilizando una frecuencia mínima de 20 Hz y estableciendo como frecuencia máxima el límite de Nyquist de la señal preparada. Debido a que la etapa anterior estableció una frecuencia de muestreo común de 16 kHz, el límite superior de la representación quedó restringido por la banda físicamente representable a esa tasa.

La conversión a Log-Mel permitió representar la energía acústica sobre una escala perceptual y comprimir su rango dinámico. El resultado fue una matriz bidimensional cuya primera dimensión corresponde a las 64 bandas Mel y cuya segunda dimensión depende del número de tramas temporales generado por la configuración STFT.

En consecuencia, cada segmento de 3 s produjo dos representaciones complementarias:

**Representación tabular**

Un vector de características acústicas.

**Representación tiempo-frecuencia**

Una matriz:

$$64 \times T$$

donde T representa el número de tramas obtenidas después de aplicar la STFT.

La notebook registró esta estructura como una representación denominada Log-Mel, con 64 bandas y una dimensión dependiente del número de tramas por segmento. Esta salida quedó destinada a los experimentos posteriores con modelos que operen directamente sobre espectrogramas.

Por tanto, la transformación completa puede representarse de la siguiente manera:

Audio original → ventana de 3 s → STFT → escala Mel → representación Log-Mel

La representación reportada utiliza 16 kHz, segmentos de 48 000 muestras, 64 bandas Mel, frecuencia mínima de 20 Hz, frecuencia máxima de 8 000 Hz, n_fft = 2048, hop_length = 512 y ventana Hann. Para reproducir exactamente la entrada aún deben explicitarse, a partir de la configuración ejecutada, la longitud efectiva de ventana, el centrado y padding de la STFT, el uso de magnitud o potencia, la convención Mel y su normalización, la conversión logarítmica o a decibelios, la referencia y el recorte de rango dinámico, el tipo numérico y la normalización posterior.

La dimensión temporal T se verificará en las matrices generadas; no se fijará por deducción mientras la política de centrado y padding permanezca sin documentar. Las notebooks de modelos profundos deberán comprobar que todas las entradas tienen dimensiones compatibles y utilizar la misma transformación durante entrenamiento e inferencia. La documentación de estos parámetros no implica que las matrices se hayan exportado; su existencia y ubicación se comprobarán en las salidas de preparación.
