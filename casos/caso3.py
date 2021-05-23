import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.decomposition import PCA
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

def caso3(dataset):
    '''
    Utilizando la totalidad de las notas, haciendo la media por mes de los SCOREs
    '''

    ###### Preprocesado de los datos
    dataset.drop(dataset[dataset.ID == 2].index, inplace=True)
    dataset.replace(np.nan, 0, inplace = True)
    dataset['MONTH'] = [x.month for x in dataset.DATE]

    # Calculamos la media
    dataset = dataset.groupby(['ID', 'CURSO'])['SCORE', 'MONTH', 'DUF'].apply(lambda x: x.groupby(['MONTH']).mean())

    # Conseguimos un df tal que cada fila representa un id
    dicc = dict()
    for id, mes, value in dataset.reset_index(drop=False)[['ID', 'MONTH', 'SCORE']].values:
        if (id, mes) in dicc.keys():
            dicc[(id, mes)] += [value]
        else:
            dicc[(id, mes)] = [value]

    # "Rellenamos" los meses que no tienen nota con 0
    for id in dataset.reset_index(drop=False)['ID'].values:
        for i in range(6):
            if (id, i+1) not in dicc.keys():
                dicc[(id, i+1)] = [0.0]

    final = {id:sum([dicc[(id, x+1)] for x in range(6)], []) for id in dataset.reset_index(drop=False)['ID'].values}
    dataset = pd.DataFrame.from_dict(final, orient='index', columns=['SCORE1', 'SCORE2', 'SCORE3', 'SCORE4', 'SCORE5', 'SCORE6']).reset_index(drop=False).rename(columns={'index':'ID'})

    # Eliminamos el SCORE1 y los alumnos con todas las notas a 0
    dataset.drop('SCORE1', axis=1, inplace = True)
    dataset.drop(dataset[(dataset.SCORE2 == 0.0) & (dataset.SCORE3 == 0.0)& (dataset.SCORE4 == 0.0)& (dataset.SCORE5 == 0.0)& (dataset.SCORE6 == 0.0)].index, inplace = True)

    # Pairplot
    dataset_test = dataset.copy(deep=True)
    sns_plot3 = sns.pairplot(dataset)
    sns_plot3.savefig("out/caso3/pairplotCaso3.png")

    # Normalizamos los datos
    min_max_scaler = preprocessing.MinMaxScaler() 
    dataset = min_max_scaler.fit_transform(dataset)

    ######### Utilización del PCA
    pca = PCA()
    pca.fit(dataset)


    plt.figure(figsize = (10,8))
    plt.plot(range(1,7), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
    plt.title("Variabilidad explicada por las componentes")
    plt.xlabel('Número de componentes')
    plt.ylabel('Variabilidad acumulativa')
    plt.savefig("out/caso3/variabPCACaso3.png")

    # Para conseguir el 80% de variabilidad, consideramos 3 componentes
    pca = PCA(n_components = 3)
    pca.fit(dataset)
    scores_pca = pca.transform(dataset)

    ########## Clasificación con k-means
    wcss = []
    for i in range(1,10):
        kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans_pca.fit(scores_pca)
        wcss.append(kmeans_pca.inertia_)

    plt.figure(figsize = (10,8))
    plt.plot(range(1,10), wcss, marker = 'o', linestyle = '--')
    plt.xlabel('Número de clusters')
    plt.ylabel('WCSS')
    plt.title('k-means con PCA')
    plt.savefig("out/caso3/kmeansCaso3.png")

    # Buscamos el número de clusters con Silhouette
    fig, ax = plt.subplots(2, 2, figsize=(15,8))
    for i in [2, 3, 4, 5]:
        '''
        Create KMeans instance for different number of clusters
        '''
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
        q, mod = divmod(i, 2)
        '''
        Create SilhouetteVisualizer instance with KMeans instance
        Fit the visualizer
        '''
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
        visualizer.fit(scores_pca)

    # Aplicamos los modelos variando el número de clusters
    modelo2PCA = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
    modelo2PCA.fit(scores_pca)

    modelo3PCA = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
    modelo3PCA.fit(scores_pca)

    modelo4PCA = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
    modelo4PCA.fit(scores_pca)

    modelo5PCA = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
    modelo5PCA.fit(scores_pca)

    # Añadimos los resultados al df (las componentes y los grupos)
    dffinal = pd.concat([dataset_test.reset_index(drop=True), pd.DataFrame(scores_pca)], axis = 1)
    dffinal.columns.values[-3: ] = ['COMPONENTE1', 'COMPONENTE2', 'COMPONENTE3']
    dffinal['GRUPO2'] = modelo2PCA.labels_
    dffinal['GRUPO3'] = modelo3PCA.labels_
    dffinal['GRUPO4'] = modelo4PCA.labels_
    dffinal['GRUPO5'] = modelo5PCA.labels_
    dffinal.to_csv("out/datasetCaso3.csv", index=False, sep=";")

    # Aplicamos un k-means de 2 cluster
    labels = dffinal.GRUPO2.values
    colores=['red','green']
    asignar=[]
    for row in labels:
        asignar.append(colores[row])
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(dffinal['COMPONENTE1'].values, dffinal['COMPONENTE2'].values, dffinal['COMPONENTE3'].values, c=asignar,s=60)
    ax.set_xlabel('COMP1')
    ax.set_ylabel('COMP2')
    ax.set_zlabel('COMP3')
    plt.savefig("out/caso3/kmeans2ClusterCaso3.png")

    # Aplicamos un k-means de 3 cluster
    labels = dffinal.GRUPO3.values
    colores=['red','green', 'blue']
    asignar=[]
    for row in labels:
        asignar.append(colores[row])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(dffinal['COMPONENTE1'].values, dffinal['COMPONENTE2'].values, dffinal['COMPONENTE3'].values, c=asignar,s=60)
    ax.set_xlabel('COMP1')
    ax.set_ylabel('COMP2')
    ax.set_zlabel('COMP3')
    plt.savefig("out/caso3/kmeans3ClusterCaso3.png")

    # Aplicamos un k-means de 4 cluster
    labels = dffinal.GRUPO4.values
    colores=['red','green','blue','cyan']
    asignar=[]
    for row in labels:
        asignar.append(colores[row])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(dffinal['COMPONENTE1'].values, dffinal['COMPONENTE2'].values, dffinal['COMPONENTE3'].values, c=asignar,s=60)
    ax.set_xlabel('COMP1')
    ax.set_ylabel('COMP2')
    ax.set_zlabel('COMP3')
    plt.savefig("out/caso3/kmeans4ClusterCaso3.png")

    # Aplicamos un k-means de 5 cluster
    labels = dffinal.GRUPO5.values
    colores=['red','green','blue','cyan', 'yellow']
    asignar=[]
    for row in labels:
        asignar.append(colores[row])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(dffinal['COMPONENTE1'].values, dffinal['COMPONENTE2'].values, dffinal['COMPONENTE3'].values, c=asignar,s=60)
    ax.set_xlabel('COMP1')
    ax.set_ylabel('COMP2')
    ax.set_zlabel('COMP3')
    plt.savefig("out/caso3/kmeans5ClusterCaso3.png")

    return dffinal