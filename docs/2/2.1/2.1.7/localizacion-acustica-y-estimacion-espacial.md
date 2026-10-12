# 2.1.7. Localización acústica y estimación espacial

La detección de una sirena responde a la pregunta de qué sonido está presente, pero una aplicación vehicular requiere además determinar de dónde procede y cómo se aproxima. Esta necesidad introduce el problema de la localización acústica.

Las técnicas basadas en arreglos de micrófonos permiten aprovechar las diferencias temporales y de fase con las que una misma onda sonora alcanza diferentes sensores. Entre las técnicas empleadas se encuentra la Diferencia de Tiempo de Llegada (TDOA), frecuentemente combinada con la Correlación Cruzada Generalizada con Transformación de Fase (GCC-PHAT). En configuraciones adecuadas, estas técnicas permiten estimar la dirección de llegada de una fuente a partir de las diferencias observadas entre los canales.

Furletov et al. (2026) desarrollaron una unidad acústica para cruces inteligentes basada en cuatro micrófonos y GCC-PHAT, orientada a la localización pasiva de fuentes en escenarios urbanos. La motivación del trabajo es precisamente complementar las tecnologías tradicionales de percepción en situaciones donde la visibilidad directa puede verse comprometida.

Banchero et al. (2025) llevaron este principio a un sistema embarcado. Su arquitectura combina análisis acústico basado en Transformers con técnicas de localización para producir información de dirección y proximidad en tiempo real. En sus experimentos, el error angular medio fue de 5,12° en el escenario simulado y de 10,30° en las mediciones reales realizadas en carretera. Los resultados nuevamente deben interpretarse según las condiciones experimentales concretas, ya que el escenario simulado y el dinámico presentan características acústicas diferentes.

La localización también evidencia una diferencia fundamental entre un clasificador convencional y un sistema de percepción acústica: mientras que la clasificación puede limitarse a una etiqueta, la localización debe mantener una relación explícita entre sonido, espacio y movimiento. En consecuencia, la señal acústica puede aportar no sólo una identificación del vehículo de emergencia, sino información espacial potencialmente útil para un sistema ADAS.
