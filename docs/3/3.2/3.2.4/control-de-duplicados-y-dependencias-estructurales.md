# 3.2.4. Control de duplicados y dependencias estructurales

Además de comprobar la integridad de los archivos, se realizó un control de duplicados mediante funciones hash SHA-256. Este procedimiento permitió identificar archivos físicamente idénticos dentro de cada corpus y determinar posibles riesgos para las futuras particiones experimentales.

En AudioSet-EV v2 se identificaron 35 archivos pertenecientes a 14 grupos de hashes repetidos; en LSSiren, 281 archivos distribuidos en 140 grupos de duplicados; y en sireNNet, 16 archivos distribuidos en ocho grupos. No se detectaron coincidencias exactas mediante SHA-256 entre datasets diferentes.

La presencia de duplicados no fue interpretada automáticamente como fuga de información. Su importancia metodológica radica en que, durante las etapas posteriores de partición de los datos, las copias pertenecientes a un mismo grupo hash no deben distribuirse entre subconjuntos independientes.

El caso de A3Siren-Recordings requiere un tratamiento adicional. La auditoría confirmó que sus 104 archivos corresponden a 13 tomas base registradas mediante ocho canales, por lo que los ocho archivos asociados a una misma toma constituyen observaciones dependientes. En consecuencia, el archivo WAV individual no se considera la unidad independiente para una eventual partición experimental; la unidad lógica será el track.

SHA-256 identifica archivos binariamente idénticos. No detecta necesariamente el mismo audio recodificado, remuestreado, recortado o sometido a aumentos de datos. En consecuencia, la ausencia de coincidencias SHA-256 entre corpus no certifica ausencia de contenido acústico compartido. Antes del entrenamiento definitivo se revisarán los identificadores de origen disponibles y los casos de similitud acústica que puedan relacionar grabaciones; las limitaciones de esa revisión se declararán expresamente.
