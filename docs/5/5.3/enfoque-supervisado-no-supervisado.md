# 5.3. Enfoque (Supervisado / No Supervisado)

El proyecto utiliza aprendizaje supervisado porque los ejemplos se asocian con una etiqueta objetivo binaria: $y\  \in \ \{ EMERGENCY,\ NON\_ EMERGENCY\}$. La variable de entrada corresponde al vector de 21 características en los modelos tradicionales y al espectrograma Log-Mel en los modelos profundos. Las diferencias de representación no modifican la definición de la tarea principal.

Las etiquetas proceden de los corpus y de las reglas de armonización registradas. Se distingue disponer de una etiqueta de archivo o track de disponer de una anotación temporal verificada para cada segmento. Cuando las etiquetas se hereden, el experimento reconocerá su carácter débil y la posibilidad de ruido de etiquetado.

Deep Learning constituye una familia de métodos dentro del aprendizaje automático y Transfer Learning una estrategia de inicialización y adaptación. La regresión logística, Random Forest, SVM, la CNN y ResNet18 pertenecen aquí al mismo experimento supervisado. El clustering y la reducción de dimensionalidad podrán utilizarse como herramientas exploratorias, pero no constituyen la regla principal de decisión.
