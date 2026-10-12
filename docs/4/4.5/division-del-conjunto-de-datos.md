# 4.5. División del Conjunto de Datos

La división del conjunto de datos tuvo como finalidad establecer subconjuntos independientes para el desarrollo y la evaluación posterior del sistema, evitando que observaciones derivadas de una misma grabación, archivos duplicados o canales pertenecientes a una misma toma fueran distribuidos entre diferentes subconjuntos experimentales. Esta consideración resulta particularmente importante debido a la existencia de segmentos temporales derivados de cada grabación y a la estructura multicanal de A3Siren-Recordings.

A diferencia de una división aleatoria aplicada directamente sobre los segmentos, el procedimiento utilizado en este trabajo realizó primero la asignación sobre la unidad independiente de origen y posteriormente propagó dicha asignación hacia los segmentos generados. De esta forma, todos los segmentos provenientes de una misma unidad conservaron el mismo subconjunto experimental.

La estrategia también incorporó un corpus externo de generalización, separado antes del desarrollo del modelo, con el objetivo de evaluar posteriormente la capacidad de transferencia frente a condiciones de captura diferentes.
