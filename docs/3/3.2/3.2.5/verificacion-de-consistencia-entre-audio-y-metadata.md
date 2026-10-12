# 3.2.5. Verificación de consistencia entre audio y metadata

Cuando los datasets proporcionaron archivos CSV asociados a los audios, se verificó la correspondencia entre los registros de metadata y los archivos físicos.

En AudioSet-EV v2 se inspeccionaron EV_Positives.csv y EV_Negatives.csv, identificándose campos relacionados con el identificador de fuente, intervalos temporales, etiquetas, tipo de segmento y estado de descarga. La existencia de numerosos registros en los CSV que no corresponden a archivos físicamente descargados fue tratada como una diferencia entre el universo de metadata disponible y el subconjunto de audio efectivamente incorporado al proyecto, en lugar de interpretarse como archivos faltantes.

En LSSiren también se verificó la relación entre sus archivos de audio y los CSV correspondientes. Durante la revisión se detectó una inconsistencia en el encabezado de Road_final.csv, que inicialmente provocaba la aparente ausencia de un registro asociado a road324.wav. La lectura alternativa del archivo permitió recuperar la estructura completa y establecer la correspondencia entre los 902 archivos de Road Noises y sus respectivos registros sin modificar el archivo original.
