# 3.2.6. Trazabilidad y reproducibilidad de la adquisición

Con el propósito de mantener la reproducibilidad del proceso, se registraron las rutas de los datasets dentro de Kaggle, la versión de Python, las principales bibliotecas utilizadas, la semilla de aleatoriedad y los archivos generados durante la auditoría. La ejecución de la auditoría utilizó Python 3.12.13 y la semilla 42, además de registrar las versiones de NumPy, pandas, Matplotlib, SoundFile y tqdm empleadas en el entorno.

Este registro permite reproducir la identificación, inspección y selección de las unidades analíticas empleadas posteriormente en el EDA. La estrategia también estableció que los archivos de entrada no fueran modificados durante la auditoría, manteniendo una separación entre datos originales, resultados de auditoría y resultados derivados del análisis exploratorio.
