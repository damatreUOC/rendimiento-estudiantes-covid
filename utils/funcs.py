'''
Archivo con todas las funciones auxiliares necesarias
'''


def comprobaciones(dataset):
    '''
    Función para la comprobación del estado correcto del dataset
    
    @dataset: El dataset a analizar
    '''
    # Los cursos deben estar comprendidos entre esos años
    if len([curso for curso in dataset.CURSO.unique() if curso not in ['17/18', '18/19', '19/20']]) > 0:
        return False

    # No puede haber una nota mayor de 10 y menor de 0
    if len([nota for nota in dataset.SCORE.unique() if (nota>10)|(nota<0) ]) > 0:
        return False

    # Los días hasta el último examen deben ser posivos
    if len([duf for duf in dataset.DUF.unique() if (duf<0) ]) > 0:
        return False

    return True