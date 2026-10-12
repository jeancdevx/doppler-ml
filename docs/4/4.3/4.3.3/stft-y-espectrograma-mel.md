# 4.3.3. STFT y espectrograma Mel

Después de la segmentación, cada ventana temporal fue transformada al dominio tiempo-frecuencia mediante la Transformada de Fourier de Tiempo Corto (STFT). Esta transformación permitió representar cómo se distribuye la energía de la señal en las diferentes frecuencias a lo largo del tiempo.

La STFT constituye una etapa intermedia de la transformación, ya que permite pasar de una representación unidimensional de la señal a una representación bidimensional en función del tiempo y la frecuencia. A partir de esta matriz espectral se realizó posteriormente la proyección sobre la escala Mel.

La elección de esta representación es coherente con el análisis exploratorio realizado previamente, en el que se utilizaron representaciones tiempo-frecuencia para inspeccionar la estructura de las señales. En esta etapa, sin embargo, la STFT deja de utilizarse únicamente con fines descriptivos y pasa a formar parte del pipeline de generación de las representaciones que serán utilizadas posteriormente por los modelos.

La transformación se aplicó sobre segmentos con una duración uniforme de 3s, por lo que las representaciones obtenidas mantienen una estructura temporal comparable entre observaciones.
