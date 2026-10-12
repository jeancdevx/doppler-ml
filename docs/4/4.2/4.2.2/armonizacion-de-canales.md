# 4.2.2. Armonización de canales

La segunda fuente de heterogeneidad correspondió al número de canales. AudioSet-EV v2 presenta señales mono, LSSiren combina grabaciones mono y estéreo, sireNNet utiliza dos canales y A3Siren-Recordings contiene archivos WAV mono organizados en grupos de ocho canales asociados a una misma toma. Esta última estructura representa una dependencia entre observaciones y no una colección de ocho grabaciones independientes.

Para la rama principal de clasificación se estableció una representación mono. En el caso de las señales estéreo, la conversión a mono se realizó únicamente en memoria durante el procesamiento, mediante una mezcla temporal de los canales, sin modificar los archivos originales. De esta forma se obtuvo una entrada homogénea para las características acústicas y las representaciones tiempo-frecuencia.

A3Siren-Recordings recibió un tratamiento específico. Aunque cada archivo individual corresponde a un canal mono, los ocho canales pertenecientes al mismo track fueron conservados como parte de la estructura multicanal de la toma. Por tanto, la armonización no eliminó la información asociada con las diferentes posiciones de captura y, simultáneamente, mantuvo el track_id como identificador de dependencia para las etapas posteriores.

| **Dataset**        | **SR original**                | **SR objetivo** | **Canales originales** | **Canales preparados** | **Normalización** |
|--------------------|--------------------------------|-----------------|------------------------|------------------------|-------------------|
| AudioSet-EV v2     | 32 kHz                         | 16 kHz          | 1                      | 1, mono                | Peak              |
| LSSiren            | 8 / 22,05 / 44,1 / 48 / 96 kHz | 16 kHz          | 1 / 2                  | 1, mono                | Peak              |
| sireNNet           | 44,1 kHz                       | 16 kHz          | 2                      | 1, mono                | Peak              |
| A3Siren-Recordings | 44,1 kHz                       | 16 kHz          | 1 por archivo          | 1 por archivo          | Peak              |

<span id="_Toc242396255" class="anchor"></span>Tabla 15. Parámetros de armonización aplicados a los corpus
