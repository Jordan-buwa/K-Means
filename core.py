import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from kmean import KMeans
from utils import datas, prep_data


def get_inertia(data, plus = False, max_k = 8):
    """
    Function to get inertia values for KMeans clustering.
    :param plus: Boolean indicating whether to use KMeans++ initialization.
    :return: List of inertia values.
    """

    inertia = []
    for i in range(1, max_k + 1):
        print(f"number of clusters: {i}")
        kmean = KMeans(n_clusters=i, init_plus=plus, max_iter=100, seed=42)
        kmean.fit(data)
        inertia.append(kmean.inertia())
    return np.array(inertia)


def get_inertia_plot(inertia, max_k = 8, plus = False):
    
    """
    Function to plot inertia values for KMeans clustering.
    :param plus: Boolean indicating whether to use KMeans++ initialization.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), inertia, marker='o')
    plt.title('Optimal k (KMeans++ initialisation)' if plus else 'Optimal k (Random initialisation)')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(range(1, max_k + 1))
    plt.grid()
    plt.show()

from sklearn.cluster import KMeans as skKMeans

def elbow_method_sklearn(X, max_k=10, init='k-means++'):
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = skKMeans(n_clusters=k, init=init, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)  # WCSS

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), wcss, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.title(f'Optimal k with {init} initialisation (Scikit-learn)')
    plt.grid(True)
    plt.show()
    return wcss
