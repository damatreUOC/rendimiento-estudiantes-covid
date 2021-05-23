'''
Archivo principal del proyecto
'''

# Importamos librerías
from utils.preparacionDatos import crear_dataset
from utils.funcs import comprobaciones, estadisticos, caracterizar_grupos 
from casos.caso1 import caso1
from casos.caso3 import caso3
from casos.caso2 import caso2

# Creamos el dataset limpio
data = crear_dataset('raw_dataset', exportar = 'Y')

# Comprobamos que se cumplen los test
assert comprobaciones(data), 'Dataset erróneo, no pasa los test'

# Llamamos a los distintos casos a considerar
dataCaso1 = caso1(data)
dataCaso2 = caso2(data)
dataCaso3 = caso3(data)


# Caracterizamos los grupos conseguidos. Como ejemplo, usamos el dataset del caso 3
caracterizar_grupos(data, dataCaso3, 'GRUPO2' )
