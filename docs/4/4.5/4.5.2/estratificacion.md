# 4.5.2. Estratificación

La partición interna reportada utiliza una proporción objetivo de validación del 20 % y semilla 42, con estratificación por etiqueta canónica binaria a nivel de la unidad de partición. La proporción se establece sobre grupos y no tiene que coincidir exactamente con la proporción de segmentos, porque las grabaciones generan cantidades diferentes de ventanas. Antes de congelar los splits se comprobará la distribución de clases, corpus, grupos y segmentos en cada subconjunto. Las relaciones de origen tienen prioridad sobre la aproximación a una proporción de clases.
