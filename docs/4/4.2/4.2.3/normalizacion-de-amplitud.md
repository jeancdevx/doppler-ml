# 4.2.3. Normalización de amplitud

Se compararon la amplitud original y la normalización por pico sobre una muestra de control de 32 unidades. El informe de preparación consigna una mediana RMS de 0,1353 y un coeficiente de variación de 1,0134 para la señal original, frente a 0,2641 y 0,4299 después de la transformación. Estos resultados describen el cambio de escala y dispersión observado en esa muestra; no demuestran por sí mismos una mejora de discriminación entre clases.

La configuración adoptada establece un pico objetivo de 0,99. La transformación modifica el nivel absoluto de cada señal, por lo que el RMS posterior se interpreta respecto de la amplitud normalizada y no como una medición calibrada de intensidad acústica, distancia o volumen original. El tratamiento de señales silenciosas y valores de pico próximos a cero deberá documentarse conforme al código ejecutado.

La preparación reporta que la proporción de muestras detectadas como clipping pasó de aproximadamente 0,00135 a cero. Esta reducción depende del umbral empleado y del cambio de escala: no demuestra que se haya corregido la saturación previamente presente en la grabación ni recuperado la información perdida. La normalización por pico se conserva como configuración inicial y su efecto predictivo podrá estudiarse mediante una comparación controlada realizada con datos de desarrollo.
