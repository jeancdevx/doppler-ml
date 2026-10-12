# 4.4.1. Características temporales

Las características temporales utilizadas fueron la raíz cuadrática media de la señal (RMS) y la tasa de cruces por cero (ZCR). Ambas fueron calculadas sobre cada segmento de 3 segundos después de las etapas de remuestreo y normalización.

El RMS resume la amplitud cuadrática media del segmento. Dado que se calcula después de la normalización por pico, describe la estructura relativa de amplitud de la señal preparada y no conserva el nivel absoluto de la grabación original. Su utilidad discriminativa se evaluará conjuntamente con las demás características.

La ZCR cuantifica la frecuencia con la que la señal cambia de signo durante el segmento. En el pipeline implementado se calculó a nivel de tramas utilizando una longitud de ventana de 2048 muestras y un hop_length de 512, y posteriormente se obtuvo el promedio para representar cada segmento mediante un único valor. Esta estrategia permite condensar en una variable la actividad de alta frecuencia y la dinámica temporal de la señal.

Estas dos características representan una dimensión energética y temporal complementaria. Sin embargo, el análisis exploratorio previo mostró que las distribuciones de RMS y ZCR pueden presentar solapamiento entre grupos, por lo que no se utilizaron como reglas independientes de clasificación.
