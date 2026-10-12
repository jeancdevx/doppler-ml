# 3.2.3. Inventario y validación de integridad

Después de incorporar los datos a Kaggle se realizó un inventario físico de los archivos de audio. La finalidad fue comprobar que los archivos disponibles correspondieran realmente a las fuentes incorporadas y determinar sus condiciones técnicas antes del análisis exploratorio.

La auditoría registró 32 429 archivos WAV distribuidos entre AudioSet-EV v2, LSSiren, sireNNet y A3Siren-Recordings. Del total, 32 427 fueron considerados utilizables, mientras que dos archivos de AudioSet-EV v2 fueron identificados como casos inválidos debido a una duración igual a cero.

| **Dataset**        | **Archivos inspeccionados** | **Utilizables** | **Inválidos** | **Observación**                   |
|--------------------|-----------------------------|-----------------|---------------|-----------------------------------|
| AudioSet-EV v2     | 28 816                      | 28 814          | 2             | Dos archivos con duración nula    |
| LSSiren            | 1 834                       | 1 834           | 0             | Sin errores de lectura            |
| sireNNet           | 1 675                       | 1 675           | 0             | Sin archivos inválidos            |
| A3Siren-Recordings | 104                         | 104             | 0             | Estructura multicanal dependiente |
| Total              | 32 429                      | 32 427          | 2             |                                   |

<span id="_Toc242396248" class="anchor"></span>Tabla 8. Resumen de integridad de los archivos de audio por dataset.
