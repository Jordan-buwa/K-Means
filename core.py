import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from kmean import KMeans
from utils import datas, prep_data


def get_inertia(data, plus = False):
    """
    Function to get inertia values for KMeans clustering.
    :param plus: Boolean indicating whether to use KMeans++ initialization.
    :return: List of inertia values.
    """

    inertia = []
    for i in range(1, 11):
        print(f"number of clusters: {i}")
        kmean = KMeans(n_clusters=i, init_plus=plus, max_iter=100, seed=42)
        kmean.fit(data)
        inertia.append(kmean.inertia())
    return np.array(inertia)


def get_inertia_plot(inertia):
    
    """
    Function to plot inertia values for KMeans clustering.
    :param plus: Boolean indicating whether to use KMeans++ initialization.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(range(1, 11))
    plt.grid()
    plt.show()

from sklearn.cluster import KMeans as skKMeans

def elbow_method_sklearn(X, max_k=10):
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = skKMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)  # WCSS

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), wcss, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.title('Elbow Method to Determine Optimal k')
    plt.grid(True)
    plt.show()
    return wcss
