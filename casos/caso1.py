import pandas as pd


def caso1(dataset):
    '''
    Agrupación de las notas en las 3 últimas y realizar el clustering
    '''
    data1 = dataset.groupby(['ID', 'CURSO'])['SCORE', 'DATE']\
                    .apply(lambda x: x.sort_values('DATE', ascending=False).head(3))\
                        .reset_index(drop=False)[['CURSO', 'SCORE', 'DUF']]



