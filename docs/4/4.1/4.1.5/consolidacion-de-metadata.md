# 4.1.5. Consolidación de metadata

Finalmente, las decisiones de limpieza se integraron en una metadata maestra que permite mantener la trazabilidad de cada unidad desde su origen hasta las etapas experimentales posteriores. Para cada observación se conservaron identificadores relacionados con el dataset de procedencia, archivo original, track cuando corresponde, etiqueta original, duración, frecuencia de muestreo, número de canales, hash y estado de limpieza.

La estructura resultante permite diferenciar entre los atributos originales del audio y los estados derivados durante la preparación. En particular, se mantienen campos destinados a identificar la pertenencia a grupos de duplicados, la relación entre archivos y tracks de A3Siren y, posteriormente, la asignación a un segmento y partición experimental.

La metadata consolidada constituye así el vínculo entre el corpus original y los datasets preparados. Esta estructura será utilizada en las siguientes etapas para registrar las transformaciones aplicadas y garantizar que los segmentos derivados puedan rastrearse hasta su unidad de origen.
