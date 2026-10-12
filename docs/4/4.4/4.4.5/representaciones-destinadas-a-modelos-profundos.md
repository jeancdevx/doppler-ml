# 4.4.5. Representaciones destinadas a modelos profundos

Paralelamente al vector tabular, cada segmento fue convertido en una representación tiempo-frecuencia mediante un espectrograma Log-Mel.

La transformación se realizó utilizando:

\- 64 bandas Mel

\- $f_{\min} = 20$ Hz

\- $f_{\max}$ igual al Nyquist de la señal preparada

\- n_fft = 2048

\- hop_length = 512

\- ventana Hann

A diferencia del vector tabular, que resume cada segmento mediante 21 valores escalares, el Log-Mel conserva la variación conjunta entre tiempo y frecuencia. Su estructura puede expresarse como:

$$X_{LogMel} \in \mathbb{R}^{64 \times T}$$

donde 64 corresponde al número de bandas Mel y T representa el número de tramas temporales resultantes de la configuración STFT.

Esta representación constituye la segunda rama de entrada preparada para el modelado y permitirá, en el Capítulo 5, comparar modelos basados en características tabulares con modelos que operen directamente sobre representaciones espectrogramáticas. La literatura revisada en el Capítulo 2 ya había documentado el uso de representaciones Log-Mel como puente entre el procesamiento digital de señales y arquitecturas convolucionales aplicadas a audio.

Debe enfatizarse que la generación de Log-Mel no implica que se haya seleccionado todavía una arquitectura de Deep Learning. En esta etapa únicamente se prepara la representación que podrá ser utilizada posteriormente por los modelos correspondientes.
