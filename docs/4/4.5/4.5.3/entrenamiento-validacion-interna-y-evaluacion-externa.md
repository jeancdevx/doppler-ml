# 4.5.3. Entrenamiento, validación interna y evaluación externa

El diseño distingue entrenamiento, validación interna fija y evaluación externa. El entrenamiento se utiliza para ajustar parámetros y desarrollar la selección de configuraciones. La validación fija se utiliza para confirmar decisiones de desarrollo y no constituye una prueba interna definitiva independiente del proceso de selección. LSSiren se utiliza como corpus externo para evaluar el modelo congelado y no participa en su ajuste predictivo.

La tabla presenta los conteos reportados por la preparación. Sus filas suman 24 379 unidades de entrenamiento y 6 095 de validación, equivalentes a 30 474 unidades de desarrollo. Este total corrige la suma textual de 30 472, pero sigue sujeto a la reconciliación del conteo de AudioSet-EV y de los grupos de origen. Por tanto, los valores se interpretan como cantidades reportadas y no como una certificación de independencia estadística.

| **Dataset**        | **Train** | **Validation** | **Generalization** |
|--------------------|-----------|----------------|--------------------|
| AudioSet-EV v2     | 23 036    | 5 758          | 0                  |
| LSSiren            | 0         | 0              | 1 693              |
| sireNNet           | 1 332     | 335            | 0                  |
| A3Siren-Recordings | 11        | 2              | 0                  |
| Total              | 24 379    | 6 095          | 1 693              |

<span id="_Toc242396258" class="anchor"></span>Tabla 18. Distribución de las unidades independientes por partición experimental

El total reportado de desarrollo es 30 474 y debe actualizarse si la reconciliación de limpieza o agrupación modifica el manifiesto. Los 11 y 2 tracks de A3Siren no deben multiplicarse por ocho para contabilizar unidades de partición. Su validación de dos tracks proporciona evidencia limitada para ese corpus y no debe presentarse como una evaluación vehicular amplia. La cantidad de archivos reservados de LSSiren puede diferir de la cantidad que produce segmentos elegibles.
