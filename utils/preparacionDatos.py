import pandas as pd


def crear_dataset(raw_dataset, exportar):
    '''
    Función que crea el dataset limpio a partir del encontrado en el artículo.

    @raw_dataset: archivo inicial
    @exportar: en caso de querer exportar el resultado deberá ser 'Y'
    
    '''
    data1920 = pd.read_excel(f'data/{raw_dataset}.xlsx', sheet_name='Applied Computing', index_col=None, na_values=['No terminado'], usecols = "A:D", skiprows=1)
    data1920.dropna(how='all', inplace=True)
    data1920.insert(loc=0, column = 'Curso', value = '19/20')

    data1819 = pd.read_excel(f'data/{raw_dataset}.xlsx', sheet_name='Applied Computing', index_col=None, na_values=['No terminado'], usecols = "E:H", skiprows=19)
    data1819.dropna(how='all', inplace=True)
    data1819.insert(loc=0, column = 'Curso', value = '18/19')

    data1718 = pd.read_excel(f'data/{raw_dataset}.xlsx', sheet_name='Applied Computing', index_col=None, na_values=['No terminado'], usecols = "I:L", skiprows=69)
    data1718.dropna(how='all', inplace=True)
    data1718.insert(loc=0, column = 'Curso', value = '17/18')

    dataComputing = pd.concat([data1718, data1819, data1920], ignore_index=True)
    dataComputing.columns = ['CURSO', 'SEQ', 'ID', 'SCORE', 'DATE']
    dataComputing['DATE'] = pd.to_datetime(dataComputing.DATE)

    # Creación de days until final
    max_dates = {curso: max(dataComputing[dataComputing.CURSO == curso].DATE) for curso in dataComputing.CURSO.unique()}
    dataComputing['DUF'] = [(max_dates[curso]-fecha).days for curso, fecha in dataComputing[['CURSO', 'DATE']].values]

    if exportar == 'Y':
        dataComputing.to_csv('out/clean_dataset.csv', sep=';', index = False)

    return dataComputing
