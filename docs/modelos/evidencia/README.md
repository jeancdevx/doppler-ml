# Procedencia de la evidencia de los informes

La fuente principal es la ejecución entregada en `notebooks/`, `data/results/` y `data/models/`, correspondiente al 12 de octubre de 2026. Esta carpeta conserva una selección documental; no sustituye al paquete completo de cada ejecución.

## Copias de las exportaciones

Los archivos bajo `preparacion/`, `*/config/`, `*/tables/`, `*/reports/` y `*/models/inference.json`, además de todas las imágenes de `../figuras/`, se copiaron byte por byte de sus fuentes. [manifest.csv](manifest.csv) registra destino, origen, tipo, tamaño y SHA-256. Se verificó igualdad entre origen y copia después de escribirlos.

Los reportes `artifact_hashes.json` son los originales: contienen rutas relativas al directorio de ejecución, no a esta selección documental. No indican que se hayan copiado todos los archivos que enumeran. Los pesos completos permanecen en `data/models/`, y las predicciones, asignaciones por segmento, búsqueda completa de umbral y logs detallados permanecen en `data/results/`. No hay nuevos scores generados con modelos en este apartado.

## Derivados para explicar el estado guardado

Los CSV de `*/inspeccion/` fueron extraídos de los modelos finales cuyos hashes coinciden con los reportes:

- `feature_parameters.csv`: medias/desviaciones de StandardScaler y, en regresión logística, coeficientes y exponenciales de coeficientes. El orden proviene de `feature_contract.json`.
- `feature_importance.csv`: `feature_importances_` del bosque final, ordenado de mayor a menor.
- `trees.csv`: profundidad, nodos y hojas de cada uno de los 300 árboles guardados.
- `normalization.csv`: 64 medias y desviaciones guardadas en cada checkpoint profundo, con bandas indexadas desde cero.
- `state_tensors.csv`: nombre, forma, dtype y número de elementos de cada tensor del `state_dict`; en ResNet incluye buffers que no son parámetros entrenables.
- `documentation_calculations.json`: métricas OOF recomputadas a umbral predeterminado y congelado, extremos de restricciones de umbral, conteos de folds/roles y resúmenes de épocas o atributos del modelo.

Estos derivados no son resultados de otro entrenamiento. Se cargaron atributos de los pipelines con joblib y tensores con `torch.load(..., weights_only=True)`. El entorno local de inspección utiliza scikit-learn 1.9.1, PyTorch 2.14.1 CPU, NumPy 2.5.4, pandas 3.0.6 y joblib 1.6.0, diferentes del Kaggle original. **La lectura de atributos no certifica que inferir con esas versiones reproduzca el modelo de scikit-learn 1.6.1.** No se ejecutó inferencia de los pipelines clásicos para presentar una recarga nueva como validada. Las métricas documentales se calculan desde scores ya exportados.

## Comprobaciones efectuadas

[validation_checks.json](validation_checks.json) registra por modelo:

- Recalcular TN, FP, FN, TP, accuracy, precisión, recall, F1 macro, FPR, ROC-AUC y AP desde `validation_predictions.csv`; coinciden con `validation_metrics.csv`.
- Repetir el recálculo por corpus y por grupo usando `max(y)` y `max(score)`; coincide con las tablas correspondientes.
- Verificar que `prediction` equivale a `score >= threshold`, que los identificadores son únicos y que no se mezclan train OOF con validación fija.
- Comprobar ausencia de LSSiren en las predicciones OOF y de validación de los cinco modelos.
- Comprobar separación de los grupos conocidos entre train y validación, y coincidencia de asignaciones CV entre los cinco modelos.
- Comprobar igualdad de la validación fija por identificador, etiqueta, corpus y grupo.
- Contrastar pesos e `inference.json` con los hashes publicados y comparar todas las configuraciones de modelos con las de resultados.

Además, se comprobó que los tres roles de cada fold profundo (`fit`, `early_stopping`, `outer_validation`) contienen grupos disjuntos y son iguales entre CNN y ResNet18. Se recorrió la tabla completa de umbrales para confirmar ausencia de puntos con recall ≥ 0,90 y FPR < 0,10. Los hashes de 22 archivos de metadata, splits, características y configuración de preparación, así como de train.npy y validation.npy, coinciden con los insumos declarados por la ejecución ResNet18. No se recomputaron las representaciones desde audio crudo ni se reentrenó.

Las pruebas de recarga de ocho ejemplos de CNN/ResNet18 son **reportes originales de Kaggle**, no pruebas nuevas de esta fase documental. Sus límites se explican en los informes. El control de hashes no prueba ausencia de duplicados perceptuales ni la verdad temporal de etiquetas. La validación estructural no evalúa generalización externa.

## Notebooks y referencias

Cada informe enlaza la notebook ejecutada y las fuentes metodológicas del proyecto. Las referencias de software apuntan a documentación oficial, preferentemente de las versiones del entorno de entrenamiento. Las fórmulas desarrollan las operaciones del proyecto; las métricas y decisiones se contrastan con las exportaciones. No se incorporaron modelos, corpus, aumentos o reglas de selección nuevos.

[Índice de los cinco informes](../README.md).
