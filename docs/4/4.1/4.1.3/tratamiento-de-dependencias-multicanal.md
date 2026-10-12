# 4.1.3. Tratamiento de dependencias multicanal

El caso de A3Siren-Recordings requirió una regla de limpieza estructural diferente a la empleada para los duplicados. El corpus contiene 104 archivos WAV organizados en 13 tomas o tracks y ocho canales por toma. Los archivos pertenecientes al mismo track corresponden a diferentes posiciones de captura durante una misma grabación, por lo que no constituyen observaciones independientes.

Por esta razón, la limpieza no eliminó los distintos canales ni los trató como duplicados. En su lugar, se estableció explícitamente la relación: track → canales asociados como una dependencia que debía conservarse durante toda la preparación.

Esta decisión es necesaria para impedir que una misma toma acústica pueda generar artificialmente múltiples observaciones independientes. Asimismo, permite conservar la información espacial potencialmente relevante de los distintos sensores para etapas posteriores del proyecto.

En consecuencia, los 104 archivos válidos de A3Siren-Recordings permanecieron disponibles, pero asociados a sus respectivos track_id y channel_id. La unidad independiente para las futuras particiones experimentales continuará siendo el track, no el archivo WAV individual. Esta consideración ya había sido establecida durante la caracterización de los datos y constituye una condición para evitar fuga de información durante la división posterior.
