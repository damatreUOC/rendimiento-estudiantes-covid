import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn import preprocessing 

def caso1(dataset):
    '''
    Agrupación de las notas en las 3 últimas y realizar el clustering
    '''
    min_max_scaler = preprocessing.MinMaxScaler() 

    dataset = dataset[pd.isnull(dataset.SCORE)==False]
    data1 = dataset.groupby(['ID', 'CURSO'])['SCORE', 'DATE', 'DUF']\
                    .apply(lambda x: x.sort_values('DATE', ascending=False).head(3))\
                        .reset_index(drop=False)[['CURSO', 'SCORE', 'DUF']]
    data_return = data1.copy()

    # Gráficos de la distribución de los datos
    fig = px.box(data1, x="CURSO", y="SCORE", color="CURSO", title="Diagrama de cajas de los resultados por cursos")
    fig.write_image("out/caso1/boxplotCaso1.png")

    fig2 = px.histogram(data1, x="SCORE", title="Distribución de las notas")
    fig2.write_image("out/caso1/histScoreCaso1.png")

    fig3 = px.histogram(data1, x="DUF", title = "Distribución de la diferencia en días hasta el último examen")
    fig3.write_image("out/caso1/histDUFCaso1.png")

    sns_plot = sns.pairplot(data1)
    sns_plot.savefig("out/caso1/pairplotCaso1.png")

    # Preparación del dataset
    data1 = pd.concat((data1, pd.get_dummies(data1["CURSO"])), axis = 1).drop("CURSO", 1)   #dummies de los cursos
    data1 = min_max_scaler.fit_transform(data1)                                             #normalizamos los datos

    # Utilizando la suma de errores cuadráticos obtenemos el número de clusters óptimos
    nc = range(1, 10)
    kmeans = [cluster.KMeans(n_clusters=i) for i in nc]
    score = [kmeans[i].fit(data1).score(data1) for i in range(len(kmeans))]

    plt.xlabel('Número de clústeres (k)')
    plt.ylabel('Suma de los errores cuadráticos')
    plt.savefig("out/caso1/codo2Caso1.png")
    plt.plot(nc,score)

    # Aplicamos el k-means con k=3
    kmeans = cluster.KMeans(n_clusters=3).fit(data1)
    
    # Asignamos el grupo a cada registro del dataset de origen
    labels = kmeans.predict(data1)
    return data_return.insert(loc=0, column = 'LABELS', value = labels)