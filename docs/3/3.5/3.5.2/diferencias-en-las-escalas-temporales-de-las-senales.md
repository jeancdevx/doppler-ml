# 3.5.2. Diferencias en las escalas temporales de las señales

El segundo hallazgo corresponde a la ausencia de una duración temporal única entre los corpus. AudioSet-EV v2 presenta una mediana de 10 segundos, LSSiren de aproximadamente 8,71 segundos, y sireNNet de 3 segundos. En A3Siren-Recordings, al utilizar el track como unidad lógica, la mediana asciende a 53 segundos, con un intervalo intercuartílico de 34 segundos.

Las diferencias son particularmente relevantes porque la duración disponible en un dataset condiciona la forma en que pueden extraerse unidades acústicas comparables. Mientras AudioSet-EV v2 y sireNNet contienen segmentos de duración prácticamente fija, LSSiren presenta una dispersión mayor y A3Siren conserva grabaciones considerablemente más largas.

Desde el punto de vista metodológico, esta característica sugiere que posteriormente deberá evaluarse una estrategia de segmentación o ventaneo temporal que permita establecer unidades de análisis compatibles con el problema de reconocimiento. No obstante, el análisis exploratorio no establece todavía una longitud de ventana determinada, ya que esa decisión deberá evaluarse durante la preparación de los datos.
