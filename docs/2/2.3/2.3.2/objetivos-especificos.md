# 2.3.2. Objetivos Específicos

Organizar y auditar los corpus acústicos seleccionados, conservando la procedencia de los archivos, las etiquetas originales y las relaciones de dependencia identificables entre grabaciones, segmentos y canales.

Preparar las señales mediante limpieza, armonización de frecuencia de muestreo y canales, normalización y segmentación, documentando las transformaciones y las limitaciones de las etiquetas disponibles.

Construir dos representaciones de entrada: un vector de 21 características acústicas por segmento y un espectrograma Log-Mel de 64 bandas, manteniendo su trazabilidad hasta la grabación de origen.

Entrenar y comparar regresión logística, Random Forest, SVM, una CNN compacta y ResNet18 con transferencia y ajuste fino, utilizando una notebook independiente para cada modelo y un protocolo común de evaluación.

Evaluar la clasificación binaria y la transferencia hacia LSSiren mediante métricas de discriminación, sensibilidad y falsos positivos, complementadas con pruebas de robustez acústica definidas antes de la evaluación final.

Integrar el modelo seleccionado en un pipeline de inferencia para un prototipo controlado y medir su costo de procesamiento, distinguiendo el tiempo de adquisición del audio, la extracción de características y la inferencia.
