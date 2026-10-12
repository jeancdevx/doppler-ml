# 4.1.1. Eliminación de archivos inválidos

La primera etapa consistió en identificar archivos que no cumplían las condiciones mínimas necesarias para su procesamiento acústico. En la auditoría realizada sobre los cuatro corpus se inspeccionaron 32 429 archivos WAV, de los cuales 32 427 fueron considerados utilizables y 2 fueron clasificados como inválidos. Ambos casos pertenecieron a AudioSet-EV v2 y presentaron duración igual a cero y ausencia de frames válidos, por lo que no podían generar una señal acústica utilizable.

Los dos archivos presentaban además el mismo contenido binario identificado mediante SHA-256, pero su exclusión se realizó principalmente por su condición de archivos inválidos. Esta distinción permite evitar que un mismo caso sea contabilizado simultáneamente como error de integridad y como duplicado experimental.

En consecuencia, los archivos inválidos fueron excluidos del conjunto analítico, pero permanecieron registrados en el cleaning_report.csv para conservar la trazabilidad del procedimiento. La eliminación fue lógica dentro del pipeline de preparación y no implicó modificar los archivos originales.
