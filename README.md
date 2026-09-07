# Sistema de Reconocimiento Acústico de Vehículos de Emergencia para Vehículos Autónomos

Resumen Ejecutivo

Este proyecto propone un sistema de reconocimiento acústico basado en técnicas de aprendizaje automático (ML) para detectar y clasificar vehículos de emergencia en aproximación mediante el sonido de sus sirenas. El objetivo es que el vehículo autónomo pueda identificar vehículos como ambulancias, bomberos o patrullas policiales por su sirena antes de que sean visibles en el campo de visión de sus cámaras, aumentando así el tiempo de reacción y mejorando la seguridad vial. En entornos urbanos con alta congestión, los retrasos en el tránsito pueden afectar críticamente la llegada oportuna de los servicios de emergencia. Nuestro sistema aborda precisamente ese problema: mediante el análisis de la señal acústica, anticipa la presencia de un vehículo de emergencia, lo que ayuda a reducir retardos en la respuesta, incrementa la eficiencia del tráfico y, en última instancia, salva vidas.

La solución integra procesamiento de señales de audio y algoritmos avanzados de ML. En la etapa de preprocesamiento se extraen características acústicas relevantes, como los Coeficientes Cepstrales en la Frecuencia de Mel (MFCC), que capturan las distribuciones de frecuencia características de cada tipo de sirena. Luego, un modelo de red neuronal convolucional residual (CNN) es entrenado sobre estas características para distinguir entre sonidos de sirenas y ruido de fondo, así como para clasificar el tipo de sirena (ambulancia, policía, bomberos). Estudios recientes han demostrado que este enfoque CNN+MFCC logra alta precisión en la clasificación de sonidos de sirenas en condiciones reales. Por ejemplo, el modelo propuesto en investigaciones similares alcanza una exactitud de clasificación muy elevada y demuestra robustez a ruido urbano variado.

Uno de los resultados clave esperados es la capacidad de detección en tiempo real. Al integrar el sistema acústico con algoritmos de control de tráfico, es posible actuar inmediatamente tras la detección de una sirena. Por ejemplo, el sistema podría conectarse con semáforos inteligentes para dar prioridad al paso del vehículo de emergencia. Esta integración dinámica de señales acústicas con el control del tráfico mejora la eficiencia general vial y acorta los tiempos de respuesta de los servicios de emergencias. A nivel de negocio y tecnología, el desarrollo de este sistema puede inscribir la empresa o proyecto en las iniciativas de ciudades inteligentes, mejorando la percepción de seguridad y abriendo posibilidades de colaboración con proveedores de soluciones de movilidad autónoma.

En resumen, el proyecto “Sistema de Reconocimiento Acústico de Vehículos de Emergencia para Vehículos Autónomos” propone un modelo ML especializado que aprovecha sensores de audio para detectar sirenas de emergencia. Al lograr clasificaciones fiables antes de la visibilidad óptica, se espera reducir colisiones, agilizar paso de emergencias y contribuir a un tráfico más eficiente y seguro. Los resultados preliminares sugieren que la combinación de características MFCC y redes CNN es eficaz y alcanza una tasa de éxito alta en escenarios complejos, validando el impacto potencial del proyecto en la movilidad urbana.

Definición del Problema

Contexto del Dominio

En entornos urbanos densamente transitados, los vehículos de emergencia (ambulancias, camiones de bomberos, patrullas policiales) deben llegar con la mayor rapidez posible ante incidentes críticos. Sin embargo, la congestión vehicular frecuente puede retrasar significativamente su avance. Estos retrasos conllevan riesgos graves, incluida la pérdida de vidas y daños materiales severos. Tradicionalmente, los conductores humanos reaccionan al sonido característico de las sirenas y a las luces de emergencia para ceder el paso. En el caso de vehículos autónomos, es esencial emular esta capacidad auditiva. Compañías como Bosch han subrayado que, a medida que crece la conducción autónoma, los vehículos necesitan “oídos” artificiales: sensores de audio con inteligencia artificial que detecten sirenas en el entorno ​(Bosch Global, s.f.)​. La legislación en muchos países exige ceder el paso a vehículos de emergencia, por lo que un automóvil autónomo debe reconocer estos sonidos y actuar en consecuencia ​(Bosch Global, s.f.)​.

Además, los sistemas actuales de asistencia al conductor y conducción autónoma se basan principalmente en cámaras, radares y LiDAR para percibir el entorno visualmente. Estas tecnologías pueden fallar en detectar vehículos emergentes que están fuera de línea de visión (por ejemplo, detrás de un edificio u obstáculo). La detección acústica complementa estos sensores: un micrófono puede “escuchar” una sirena que aún no se ve ​(Bosch Global, s.f.)​. De hecho, investigaciones recientes han demostrado que sensores acústicos inteligentes pueden identificar vehículos de emergencia antes de que sean visibles por cámaras. Este reconocimiento anticipado es crucial para ajustar la trayectoria o velocidad del vehículo autónomo con suficiente anticipación ​(Bosch Global, s.f.)​.

En términos de aprendizaje automático, este problema entra en el dominio de la detección y clasificación de eventos sonoros (audio event detection) en entornos ruidosos. Requiere distinguir patrones acústicos de sirena de una amplia variedad de ruidos urbanos (tráfico, claxon, obras, etc.). Técnicas basadas en redes neuronales convolucionales (CNN) han avanzado notablemente en tareas similares, gracias a su robustez frente a ruido y su capacidad para extraer características invariantes de los espectrogramas sonoros ​(Humphrey, Bello, & LeCun, 2013)​ ​(Piczak, 2015)​. Así, el contexto del dominio comprende el reconocimiento de señales acústicas críticas en tiempo real dentro del marco de sistemas de transporte inteligente y la conducción autónoma ​(Virtanen, Ellis, & Plumbley, 2018)​.

Descripción del Problema

El problema específico que aborda este proyecto es diseñar un sistema de detección acústica en tiempo real capaz de identificar la aproximación de un vehículo de emergencia a partir del sonido de su sirena y, a partir de ello, clasificar el tipo de emergencia (ambulancia, policía, bombero). Es decir, dado un flujo de audio capturado por micrófonos del vehículo autónomo, el sistema debe señalar si en él está presente la sirena de un vehículo de emergencia y determinar su categoría. Esto implica varios subproblemas: primero, la detección binaria de “sirena vs ruido ambiental”; y segundo, la clasificación multicategoría de la sirena según el tipo de vehículo ​(Mesaros, Heittola, & Virtanen, 2016)​.

A nivel técnico, el reto radica en la alta variabilidad y complejidad de los datos de audio. Las sirenas pueden tener diferentes tonos (dos tonos, lamento, yelp, etc.) dependiendo del país y el tipo de vehículo ​(Zhong, Gu, Prieto, & Fehler, 2025)​. Además, el audio urbano es muy ruidoso y no estacionario. Estudios señalan que la diversidad global de las señales de sirena es amplia, lo que exige que los modelos generalicen bien a datos no vistos ​(Mittal & Chawla, 2023)​. Por lo tanto, el sistema debe ser robusto ante variaciones de volumen, reverberación, y ruido de fondo (tráfico, música, sirenas ajenas) ​(Zhong, Gu, Prieto, & Fehler, 2025)​​ (Mittal & Chawla, 2023)​.

Para la extracción de características se utilizarán técnicas de procesamiento de señales como la transformada de Fourier de corto plazo (STFT) o directamente la extracción de MFCC, que convierte cada segmento de audio en un vector de características espectrales que capturan la esencia de la sirena. Estas características alimentan posteriormente a un modelo supervisado de ML (por ejemplo, una CNN) entrenado con ejemplos etiquetados de sirenas y ruido. La literatura más reciente recomienda arquitecturas CNN residuales para este fin, que logran alta precisión en la clasificación de sirenas. La detección debe realizarse en milisegundos, ya que cualquier demora puede impactar el tiempo de reacción del vehículo.

En síntesis, el problema consiste en anticipar la presencia de vehículos de emergencia mediante el reconocimiento acústico de sus sirenas y clasificar su tipo, todo dentro de las restricciones de un entorno ruidoso y en tiempo real. Al resolverlo, se colabora con mejorar la seguridad vial de los vehículos autónomos y su alineación con requisitos legales y de eficiencia del tráfico.

Objetivos Específicos

Los objetivos específicos del proyecto se definen en torno a la detección acústica y el aprendizaje automático:

Detectar señales de sirena en tiempo real: Desarrollar un módulo que, a partir de la entrada de audio del entorno, identifique la presencia de una sirena de emergencia en circulación.

Clasificar el tipo de vehículo de emergencia: Diferenciar las sirenas detectadas en categorías (p.ej., ambulancia, bombero, policía) basándose en sus patrones acústicos característicos.

Procesamiento de audio y extracción de características: Implementar técnicas de preprocesamiento (filtrado de ruido, normalización) y extraer características relevantes (MFCC, espectrogramas, etc.) que permitan distinguir las sirenas del ruido de tráfico.

Construir y entrenar un modelo de ML eficaz: Evaluar varias arquitecturas de aprendizaje supervisado (CNNs, RNNs, etc.) para seleccionar y entrenar aquella que ofrezca mejor desempeño en la tarea. Los trabajos previos sugieren el uso de CNN residuales dado su alto rendimiento en clasificación de audio.

Integración en sistema autónomo: Diseñar el sistema para que pueda integrarse con otros módulos del vehículo (sensores de cámara, actuadores de frenado). Por ejemplo, en caso de detección positiva, el sistema enviaría señales al controlador del vehículo o al sistema de semáforos inteligentes.

Robustez y prueba en el mundo real: Validar el sistema con datos reales grabados en distintos escenarios urbanos, asegurando alta precisión aun en presencia de ruido, y aplicando técnicas de validación cruzada para comprobar la generalización.

Criterios de Éxito

Para evaluar el éxito del sistema, se establecen criterios cuantitativos y cualitativos claros:

Precisión (Accuracy) de Clasificación: El modelo debe alcanzar una precisión alta (idealmente superior al 90–95%) al distinguir entre audio de sirena y ruido. Esto se medirá tanto en condiciones ideales como en escenarios urbanos reales.

Recall (Sensibilidad): Es crítico detectar la mayoría de las emergencias. El recall (tasa de verdaderos positivos) debe ser muy alta, minimizando las falsas alarmas. Un falso negativo (no detectar una sirena) podría tener consecuencias graves, por lo que se priorizará minimizar estos casos.

Latencia de Detección: El sistema debe operar en tiempo real. El retardo entre la captura del audio y la respuesta del modelo debe ser inferior a un umbral aceptable (por ejemplo, <250 ms), para permitir reacciones oportunas.

Robustez ante Ruido: El desempeño no debe degradarse sustancialmente en presencia de ruido de fondo intenso. Se medirá la performance en grabaciones con tráfico, música de ciudad y otras sirenas simultáneas para validar la resiliencia.

Exactitud de Clasificación de Tipo: Además de detectar sirenas, el modelo debe clasificar correctamente el tipo de vehículo de emergencia. Se evaluará la matriz de confusión para asegurar que, por ejemplo, una sirena de ambulancia no se confunda con una policial.

Cumplimiento de Normativas: El sistema respetará estándares de privacidad y seguridad (p.ej., no grabar ni almacenar conversaciones) y cumplirá con regulaciones locales sobre sistemas de asistencia vehicular.

Facilidad de Integración y Escalabilidad: A nivel técnico, el sistema se considerará exitoso si su diseño modular permite integrarlo en entornos reales (APIs estándar, compatibilidad con hardware automotriz) y escalar a diversos modelos de vehículos.

El cumplimiento de estos criterios se determinará mediante pruebas controladas y ensayos en campo. Por ejemplo, estudios similares reportan métricas de rendimiento elevadas usando CNNs con MFCC, y la solución diseñada deberá alcanzar niveles comparables. En conjunto, estos criterios aseguran que el sistema no solo resuelva el problema teórico de clasificación de audio, sino que sea práctico y efectivo en el contexto real de la conducción autónoma.
