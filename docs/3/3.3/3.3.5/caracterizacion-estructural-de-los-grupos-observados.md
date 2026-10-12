# 3.3.5. Caracterización estructural de los grupos observados

El análisis exploratorio de las categorías permitió identificar diferentes niveles de organización entre los datasets. AudioSet-EV v2 presenta una separación general entre Positive_files y Negative_files, pero mantiene adicionalmente información procedente de la ontología de AudioSet. La metadata de positive_labels contiene miles de combinaciones distintas, por lo que estas etiquetas no fueron reinterpretadas durante la exploración como categorías exclusivas de ambulancia, policía o bomberos.

LSSiren presenta una organización binaria definida mediante los grupos Ambulance y Road, mientras que sireNNet dispone de cuatro categorías estructuradas mediante directorios: ambulance, firetruck, police y traffic. En A3Siren-Recordings, la organización Siren/Noise se encuentra además asociada a tracks y canales, por lo que la estructura del corpus representa simultáneamente una categoría acústica y una condición de captura multicanal.

Esta diversidad de estructuras implica que las cantidades observadas no pueden interpretarse únicamente como un problema de balance entre clases. También es necesario considerar la naturaleza de la unidad analítica y la procedencia de las etiquetas. Este aspecto será especialmente importante durante la preparación de los datos y la posterior definición de las particiones experimentales.
