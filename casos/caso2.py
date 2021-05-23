import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from mpl_toolkits.mplot3d import Axes3D
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import KMeans

def caso2(dataset):
    '''
    Utilizando la media de cada estudiante de los 3 últimos meses
    '''

    ######## Preprocesado
    dataset = dataset.replace(np.nan, 0)
    dataset['MONTH'] = [x.month for x in dataset.DATE]
    dataset = dataset.groupby(['ID', 'CURSO'])['SCORE', 'MONTH', 'DUF'].apply(lambda x: x.groupby(['MONTH']).mean().head(3))

    # Conseguimos un df tal que cada fila representa un id
    dicc = dict()
    for id, value in dataset.reset_index(drop=False)[['ID', 'SCORE']].values:
        if id != 2:
            if id in dicc.keys():
                dicc[id] += [value]
            else:
                dicc[id] = [value]

    for i in dicc.keys():
        while len(dicc[i]) < 3:
            dicc[i] += [0]

    dataset = pd.DataFrame.from_dict(dicc, orient='index', columns=['SCORE1', 'SCORE2', 'SCORE3'])
    dataset_test = dataset.copy(deep=True)


    ######## k-means

    # Pairplot del dataset
    sns_plot2 = sns.pairplot(dataset)
    sns_plot2.savefig("out/caso1/pairplotCaso2.png")

    # Normalizamos los datos
    min_max_scaler = preprocessing.MinMaxScaler() 
    dataset = min_max_scaler.fit_transform(dataset)

    # Gráfico 3D para la dispersión
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2])
    plt.savefig("out/caso2/dispersionCaso2.png")

    # Utilizando la suma de errores cuadráticos, extraemos el número de clusters
    nc = range(1, 10)
    kmeans = [cluster.KMeans(n_clusters=i) for i in nc]
    score = [kmeans[i].fit(dataset).score(dataset) for i in range(len(kmeans))]

    plt.xlabel('Número de clústeres (k)')
    plt.ylabel('Suma de los errores cuadráticos')
    plt.savefig("out/caso1/codo2Caso1.png")
    plt.plot(nc,score)
    plt.savefig("out/caso2/elbowCaso2.png")


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
        visualizer.fit(dataset)

    # Aplicamos un k-means de 5 cluster
    kmeans = cluster.KMeans(n_clusters=5).fit(dataset)
    centroids = kmeans.cluster_centers_

    labels = kmeans.predict(dataset)
    colores=['red','green','blue','cyan', 'yellow']
    asignar=[]
    for row in labels:
        asignar.append(colores[row])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=asignar,s=60)
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', c='black')
    ax.set_xlabel('SCORE1')
    ax.set_ylabel('SCORE2')
    ax.set_zlabel('SCORE3')
    plt.savefig("out/caso2/finalCaso2.png")
    
    dataset_test['GRUPO'] = kmeans.labels_
    return dataset_test














        