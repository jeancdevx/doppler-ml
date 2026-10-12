# 4.3.5. Aumento de datos

Como parte del diseño inicial de la preparación se contempló la posibilidad de utilizar transformaciones de aumento de datos para introducir variabilidad artificial en las señales, particularmente mediante perturbaciones asociadas a amplitud, ruido o modificaciones temporales.

Sin embargo, en la ejecución final de la notebook esta estrategia se mantuvo desactivada. En consecuencia, el conjunto preparado utilizado como base para el desarrollo del modelo permanece compuesto únicamente por segmentos obtenidos de las señales originales mediante las transformaciones de preparación descritas anteriormente. No se generaron versiones sintéticas adicionales de los segmentos.

Esta decisión permite que las etapas posteriores de selección y entrenamiento comiencen sobre una base de datos exclusivamente derivada de las grabaciones disponibles, manteniendo claramente diferenciadas las transformaciones deterministas de preparación y cualquier eventual estrategia de aumento que pueda ser evaluada posteriormente.

En caso de incorporar aumento de datos en una etapa posterior, la configuración de la notebook establece que este procedimiento deberá aplicarse exclusivamente sobre el conjunto de entrenamiento y nunca sobre validation ni sobre el corpus de generalización.
