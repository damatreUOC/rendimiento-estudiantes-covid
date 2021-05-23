'''
Archivo con todas las funciones auxiliares necesarias
'''

import pandas as pd

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

def estadisticos(df, scores):
    a = df.mean()[scores].values.tolist()
    a.extend(df.std()[scores].values)
    a.append(df[scores].mean().mean())
    a.append(df[scores].std().std())
    a.extend(df.median()[scores].values)
    a.append(df[scores].median().median())
    a.append(df.MEDIADUF.mean())
    a.append(df[df.CURSO == '17/18'].ID.count())
    a.append(df[df.CURSO == '18/19'].ID.count())
    a.append(df[df.CURSO == '19/20'].ID.count())
    return [round(d, 2) for d in a]


def caracterizar_grupos(dataComputing, dataset, GRUPO):
    cursos_por_est = {id: curso for id, curso in dataComputing[['ID', 'CURSO']].values}
    mediaDUF = {id: dataComputing.loc[dataComputing.ID == id, 'DUF'].mean() for id in dataComputing['ID'].values}
    dataset['CURSO'] = [cursos_por_est[id] for id in dataset['ID'].values]
    dataset['MEDIADUF'] = [mediaDUF[id] for id in dataset['ID'].values]
    scores = dataset.columns[dataset.columns.str.contains('SCORE')]
    columnas = ['MEDIA_SCORE2', 'MEDIA_SCORE3', 'MEDIA_SCORE4', 'MEDIA_SCORE5','MEDIA_SCORE6', 
            'SD_SCORE2', 'SD_SCORE3', 'SD_SCORE4', 'SD_SCORE5', 'SD_SCORE6', 'MEDIA', 'SD',
            'MEDIANA_SCORE2', 'MEDIANA_SCORE3', 'MEDIANA_SCORE4', 'MEDIANA_SCORE5','MEDIANA_SCORE6',
            'MEDIANA','MEDIA_DUF','N_CURSO1','N_CURSO2','N_CURSO3']
    
    dic = dict()
    for i in dataset[GRUPO].unique():
        dic['GRUPO{}'.format(i)] = estadisticos(dataset[dataset[GRUPO] == i], scores)
    
    return pd.DataFrame.from_dict(dic, orient='index', columns=columnas)