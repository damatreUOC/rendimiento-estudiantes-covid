'''
Archivo principal del proyecto
'''

# Importamos librerías
import pandas as pd
from utils.preparacionDatos import crear_dataset
from utils.funcs import comprobaciones

# Creamos el dataset limpio
data = crear_dataset('raw_dataset', exportar = 'Y')

# Comprobamos que se cumplen los test
assert comprobaciones(data), 'Dataset erróneo, no pasa los test'






