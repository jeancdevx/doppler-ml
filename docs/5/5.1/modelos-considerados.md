# 5.1. Modelos Considerados

Se compararán cinco modelos de aprendizaje supervisado para la tarea binaria, utilizando dos representaciones de audio. La comparación busca establecer el desempeño obtenido con descriptores explícitos frente al aprendizaje sobre espectrogramas, considerando también el costo computacional y la transferencia entre corpus. Los antecedentes de la literatura orientan la elección de candidatos, pero no establecen cuál tendrá el mejor resultado en los datos del proyecto.

Los modelos tradicionales serán regresión logística, Random Forest y SVM sobre las 21 características acústicas. La regresión logística proporcionará una referencia lineal; Random Forest permitirá estudiar relaciones no lineales mediante un ensamble de árboles; y SVM permitirá evaluar una frontera de margen. El kernel y el presupuesto de SVM se definirán según la factibilidad computacional, evitando asumir que una búsqueda extensa sobre todos los segmentos será viable. Las transformaciones aprendidas se ajustarán dentro del entrenamiento de cada fold.

Los modelos profundos serán una CNN compacta entrenada desde cero y ResNet18 inicializada con pesos de un dominio visual y adaptada a la clasificación binaria. Ambas utilizarán Log-Mel. La CNN permitirá establecer una referencia profunda sin preentrenamiento; ResNet18 permitirá evaluar transferencia y posterior ajuste fino. Transfer Learning se entiende como estrategia de inicialización y adaptación dentro del aprendizaje supervisado, no como un paradigma separado de Machine Learning o Deep Learning.

Para ResNet18 se propone utilizar pesos preentrenados en ImageNet como forma de evitar el solapamiento específico que podría producirse al inicializar con modelos entrenados en AudioSet. Esta elección no prueba ausencia de toda relación entre fuentes ni garantiza que la transferencia visual sea beneficiosa para audio. Los pesos, la adaptación de canales, la normalización de entrada, las capas congeladas y la política de ajuste fino se explicitarán antes de ejecutar el experimento.

| **Familia**      | **Modelo**          | **Representación de entrada** | **Estrategia**                  | **Función dentro del experimento** |
|------------------|---------------------|-------------------------------|---------------------------------|------------------------------------|
| Machine Learning | Regresión Logística | 21 características            | Entrenamiento supervisado       | Línea base lineal                  |
| Machine Learning | Random Forest       | 21 características            | Entrenamiento supervisado       | Línea base no lineal               |
| Machine Learning | SVM                 | 21 características            | Entrenamiento supervisado       | Línea base no lineal de margen     |
| Deep Learning    | CNN compacta        | Log-Mel 64 × T                | Entrenamiento desde cero        | Línea base profunda                |
| Deep Learning    | ResNet18            | Log-Mel 64 × T                | Transfer Learning + fine-tuning | Modelo principal                   |

<span id="_Toc242396259" class="anchor"></span>Tabla 19. Modelos considerados para la clasificación acústica
