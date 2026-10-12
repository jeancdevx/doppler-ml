# 4.1.2. Control de duplicados

Una vez excluidos los archivos inválidos, se realizó un control de duplicación mediante el cálculo y comparación de hashes SHA-256. El objetivo fue evitar que múltiples copias binarias de un mismo archivo pudieran aparecer posteriormente como observaciones distintas y afectar la partición experimental.

La auditoría inicial identificó 35 archivos distribuidos en 14 grupos de hash repetido en AudioSet-EV v2, 281 archivos distribuidos en 140 grupos en LSSiren y 16 archivos distribuidos en 8 grupos en sireNNet. No se identificaron coincidencias exactas mediante SHA-256 entre datasets diferentes.

La reducción del inventario distingue archivos inválidos de copias redundantes. Los grupos SHA-256 se identificaron sobre el inventario inicial, que incluía dos archivos inválidos de AudioSet-EV v2 con el mismo contenido binario. Para evitar una doble contabilización, el número de copias redundantes excluidas después de retirar inválidos debe obtenerse del reporte de limpieza. El informe inicial consigna 21 copias excedentes en AudioSet-EV v2, 141 en LSSiren y 8 en sireNNet; el primer valor debe reconciliarse con el universo sobre el que se calcularon los hashes y con el conteo final del corpus.

A3Siren-Recordings no presentó duplicados exactos mediante SHA-256; sin embargo, su estructura multicanal requiere un tratamiento diferente, puesto que los ocho archivos correspondientes a una misma toma no representan ocho grabaciones independientes.

| **Dataset**        | **Inicial** | **Inválidos** | **Duplicados** | **Final** |
|--------------------|-------------|---------------|----------------|-----------|
| AudioSet-EV v2     | 28 816      | 2             | 21             | 28 794    |
| LSSiren            | 1 834       | 0             | 141            | 1 693     |
| sireNNet           | 1 675       | 0             | 8              | 1 667     |
| A3Siren-Recordings | 104         | 0             | 0              | 104       |

<span id="_Toc242396254" class="anchor"></span>Tabla 14. Resultado de la limpieza de los datos
