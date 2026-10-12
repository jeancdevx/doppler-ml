# 3.5.5. Dependencia de las condiciones de captura

El quinto hallazgo evidencia que las condiciones de captura constituyen una fuente adicional de variabilidad. En LSSiren se observaron cinco frecuencias de muestreo y configuraciones mono y estéreo, mientras que A3Siren-Recordings presenta una misma toma distribuida en ocho canales dependientes. En A3Siren, la mediana del rango de RMS entre canales de una misma toma fue de 0,0088, lo que evidencia variación entre las diferentes observaciones de canal pertenecientes al mismo track.

La importancia de A3Siren radica particularmente en que sus canales no representan grabaciones independientes. Los ocho archivos correspondientes a un track constituyen distintas observaciones de una misma toma acústica, por lo que tratarlos como ocho muestras independientes podría inflar artificialmente la cantidad de observaciones disponibles y afectar posteriormente la estimación del desempeño del modelo.

![Figura 16. Mediana e intervalo intercuartílico del centroide espectral.](../../../img/figura-16-mediana-e-intervalo-intercuartilico-del-centroide-espectral.png)

<span id="_Toc242396275" class="anchor"></span>Figura 16. Mediana e intervalo intercuartílico del centroide espectral.
