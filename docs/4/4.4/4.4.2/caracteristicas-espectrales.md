# 4.4.2. Características espectrales

El segundo grupo estuvo constituido por características derivadas de la representación espectral de cada segmento. Se utilizaron:

\- frecuencia dominante

\- centroide espectral

\- ancho de banda espectral

\- roll-off espectral

\- flujo espectral

\- flatness espectral

La representación espectral se obtuvo mediante STFT con n_fft = 2048, hop_length = 512 y ventana Hann, configuración mantenida de forma consistente con las transformaciones definidas en la etapa anterior.

La frecuencia dominante se definió a partir del componente frecuencial con mayor magnitud media dentro del segmento, excluyendo la componente de frecuencia cero. Este descriptor permite localizar la región frecuencial con mayor concentración de magnitud.

El centroide espectral resume la distribución de energía de la representación frecuencial alrededor de su centro de masa. El ancho de banda espectral complementa esta información al describir la dispersión del contenido frecuencial respecto del centroide.

El roll-off espectral identifica la frecuencia por debajo de la cual se acumula el 85 % de la magnitud o potencia espectral utilizada en el cálculo. La definición concreta depende del espectro suministrado por la implementación y deberá registrarse junto con la configuración de extracción. Este descriptor resume la distribución espectral y no identifica por sí solo la clase acústica.

El flujo espectral fue incorporado para caracterizar la variación de la distribución frecuencial entre tramas consecutivas, mientras que la planitud espectral (spectral flatness) se utilizó como descriptor de la concentración relativa frente a una distribución espectral más uniforme.
