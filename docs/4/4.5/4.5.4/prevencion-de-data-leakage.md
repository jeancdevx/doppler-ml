# 4.5.4. Prevención de data leakage

La preparación documenta controles para conservar en una misma partición las ventanas de un archivo, los canales de un track de A3Siren y las copias exactas identificadas mediante SHA-256. Estos controles reducen riesgos conocidos de fuga, pero no certifican ausencia de relaciones acústicas entre archivos con hashes diferentes.

Antes del entrenamiento definitivo se comprobará que no existan intersecciones de identificadores de origen entre particiones, que las asignaciones sean consistentes para todos los derivados y que las etiquetas correspondan a las reglas canónicas documentadas. Se revisarán las variantes aumentadas y los identificadores de video disponibles. Los casos sin procedencia recuperable se informarán como limitaciones. El escalamiento, la selección de características, los pesos de clase y cualquier transformación aprendida se ajustarán exclusivamente con el entrenamiento correspondiente a cada experimento o fold.

LSSiren no se utilizará para elegir hiperparámetros, épocas, umbrales ni modelos a partir de sus resultados predictivos. Si después de evaluarlo se modifica el sistema utilizando sus errores, esa ejecución dejará de ser una evaluación final intacta y deberá identificarse como un nuevo ciclo de desarrollo que requiere otra prueba independiente.
