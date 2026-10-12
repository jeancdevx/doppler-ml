# 4.3.2. Ventaneo y solapamiento

Las etiquetas disponibles describen archivos o tracks según el corpus. Su propagación a ventanas de tres segundos constituye un supuesto de supervisión débil cuando no existen anotaciones temporales que confirmen la presencia del evento en cada ventana. En particular, un archivo positivo puede contener intervalos sin sirena. La construcción de etiquetas deberá documentar si se hereda la clase del archivo, se dispone de anotaciones temporales o se realiza una revisión adicional. Las métricas obtenidas con etiquetas heredadas evalúan la concordancia con esas etiquetas y no certifican una localización temporal exacta del evento.

Una vez establecida la duración de la ventana, se definió un esquema de ventaneo con 50% de solapamiento. Esto corresponde a un hop temporal de 1,5s para una ventana de 3,0s. La configuración permite que dos segmentos consecutivos compartan la mitad de su contenido temporal, proporcionando una cobertura continua de las señales y evitando que una característica localizada en el límite de una ventana quede completamente excluida.

El procedimiento empleado puede expresarse como:

señal original → ventana de 3,0 s → desplazamiento de 1,5 s → nueva ventana

Por tanto:

$$overlap = 1 - \frac{hop}{window}$$

y con los parámetros utilizados:

$$overlap = 1 - \frac{1.5}{3.0} = 0.50$$

equivalente a un 50 % de solapamiento.

La configuración fue aplicada de manera uniforme a las unidades analizadas, manteniendo asociada cada ventana con el identificador de origen correspondiente. Esta relación resulta necesaria para evitar que ventanas provenientes de la misma grabación puedan ser consideradas independientes durante la división del conjunto de datos.
