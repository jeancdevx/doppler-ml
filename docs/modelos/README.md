# Documentación detallada de los modelos entrenados

Estos cinco informes independientes documentan las ejecuciones entregadas del **12 de octubre de 2026**. Cada uno desarrolla el recorrido completo desde audio y etiquetas hasta arquitectura, matemática, folds, búsqueda, umbral, métricas, gráficas, costo y recarga del artefacto.

- [Regresión logística](01_regresion_logistica.md): 21 características, escalamiento, pérdida logística, L2, pesos de clase y coeficientes del modelo final.
- [Random Forest](02_random_forest.md): bootstrap, Gini, balance por árbol, 300 árboles, tamaño real e importancias por impureza.
- [SVM](03_svm.md): comparación lineal/Nyström, kernel RBF aproximado de 256 componentes, pérdida squared hinge y margen sin calibrar.
- [CNN](04_cnn.md): Log-Mel, tres bloques convolucionales, GroupNorm, 23 473 parámetros, entrenamiento y parada interna.
- [ResNet18](05_resnet18.md): inicialización ImageNet, bloques residuales, cabeza binaria, congelamiento, BatchNorm y ajuste fino de 11 177 025 parámetros.

## Resultados comparables de validación interna fija

Todos se evalúan sobre los mismos **29 315 segmentos**, con **8 137 positivos**, **21 178 negativos** y **6 090 grupos**. Train contiene **118 314 segmentos**. Los tres folds externos y, entre redes, los subconjuntos de parada son iguales. Cada modelo conserva su propio umbral escogido con OOF; la SVM usa margen y los otros scores en [0,1], así que los valores de umbral no son comparables como probabilidades universales.

| Modelo e informe | F1 macro | Recall emergencia | FPR | ROC-AUC | AP | Umbral |
| --- | --- | --- | --- | --- | --- | --- |
| [Regresión logística](01_regresion_logistica.md) | 0,798429 | 0,686002 | 0,099254 | 0,881569 | 0,774164 | 0,625500 |
| [Random Forest](02_random_forest.md) | 0,827919 | 0,716972 | 0,077392 | 0,911667 | 0,836286 | 0,508753 |
| [SVM](03_svm.md) | 0,818503 | 0,710458 | 0,086127 | 0,904343 | 0,821260 | 0,227037 |
| [CNN](04_cnn.md) | 0,892269 | 0,792430 | 0,035461 | 0,956316 | 0,926079 | 0,528290 |
| [ResNet18](05_resnet18.md) | 0,905709 | 0,822662 | 0,033573 | 0,962987 | 0,930496 | 0,654784 |

ResNet18 tiene el mayor F1 macro global, seguido de CNN. **Ningún modelo cumple simultáneamente recall ≥ 0,90 y FPR < 0,10 por segmento en validación fija**, y ninguno encontró un umbral OOF viable bajo ambas restricciones. Todos los FPR globales de validación quedan por debajo de 0,10; el recall es la restricción que falla allí. Se aplicó el fallback de máximo F1 macro OOF previsto en el código. Estos resultados no demuestran cumplimiento externo.

El promedio global está dominado por **AudioSet-EV v2 (97,663 % de los segmentos de validación)**. El desglose por corpus es indispensable: A3 contiene solo **dos tracks**, uno de cada clase, y los cinco modelos tienen un desempeño deficiente en sus segmentos. ResNet18, aunque lidera globalmente, presenta **FPR 0,939103 en A3**. Los informes conservan estas diferencias en tablas y conteos; no se han reducido a una única cifra de ranking.

Hay intervalos de bootstrap de grupos con 500 repeticiones en cada informe. La incertidumbre es condicional al modelo y umbral fijados; no se ejecutó una prueba pareada entre modelos ni un bootstrap de todo el proceso de selección. El orden de las estimaciones puntuales no equivale a significación estadística ni a una decisión de despliegue.

## Costo observado y límites del benchmark

| Modelo | Entorno | Búsqueda CV, s | Ajuste final, s | p50, ms | p95, ms | Modelo final, MiB |
| --- | --- | --- | --- | --- | --- | --- |
| Regresión logística | CPU | 9,573 | 0,940 | 0,460 | 0,620 | 0,001 |
| Random Forest | CPU | 1042,371 | 177,774 | 65,707 | 67,690 | 86,393 |
| SVM | CPU | 55,573 | 23,309 | 14,699 | 15,605 | 0,261 |
| CNN | 2 × T4; latencia en 1 GPU | 1859,606 | 423,217 | 0,967 | 1,615 | 0,096 |
| ResNet18 | 2 × T4; latencia en 1 GPU | 1818,528 | 249,220 | 3,402 | 3,578 | 42,706 |

Los tiempos proceden de las exportaciones. La latencia clásica incluye el pipeline sobre vectores ya calculados; la profunda incluye el forward sobre tensores ya normalizados y residentes en GPU, con AMP. **No incluyen preparación de audio ni latencia completa de alerta**. CPU y GPU no constituyen una comparación a igualdad de hardware. Los timers de búsqueda y ajuste no se suman nuevamente a los tiempos de sus folds o épocas, y hay verificaciones anidadas en lectura de redes.

## Evidencia conservada y comprobaciones

Las gráficas originales PNG/SVG y tablas/configuraciones clave se copiaron a este apartado para poder consultarlas en GitHub, ya que `data/results/` y `data/models/` están ignorados por Git. No se duplicaron los modelos binarios, matrices de varios GiB, predicciones completas ni eventos TensorBoard.

Se comprobaron hashes de los cinco modelos, archivos de inferencia y configuraciones; coincidencia de folds y validación; exclusión predictiva de LSSiren; separación de grupos; separación interna de las redes; y equivalencia de métricas globales, por corpus y por grupo al recalcularlas desde predicciones existentes. Se contrastaron además los hashes de insumos de preparación consumidos, incluidas las matrices train y validation. [Procedencia y verificación](evidencia/README.md), [comprobaciones](evidencia/validation_checks.json) y [manifiesto de copias](evidencia/manifest.csv).

Se documentaron dos detalles de evidencia sin alterar sus fuentes: el snapshot completo de preparación guarda `RUNNING` porque se exporta antes del cambio final de estado, aunque READY y validaciones están presentes; y el piso de planitud descrito en `feature_definitions.json` difiere del valor por defecto realmente usado por Librosa 0.11.0. Los informes explican las convenciones efectivas y conservan el original.

**Evaluación externa de LSSiren, robustez y prototipo permanecen pendientes.** La documentación histórica en los puntos 1–5 se conserva; estos informes identifican la nueva ejecución y sus cifras, sin atribuirlas retrospectivamente a otros experimentos.

[Volver al índice del proyecto](../README.md).
