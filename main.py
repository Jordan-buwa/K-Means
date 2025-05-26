from kmean import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import datas, prep_data

if __name__ == "__main__":

    # Loading and preprocessing the data
    data = datas()
    print(data.head())
    x_mall = prep_data(data)

    # Clustering the data using KMeans
    kmean = KMeans(n_clusters=5, init_plus=False, max_iter=100)
    kmean.fit(x_mall)
    # Plotting the clusters
    kmean.plot()

    # Clustering the data using KMeans
    kmean = KMeans(n_clusters=5, init_plus=True, max_iter=100, seed=42)
    kmean.fit(x_mall)
    # Plotting the clusters
    kmean.plot()

    # Finding the optimal number of clusters using the Elbow method (KMeans++ initialization)
    inertia = []
    for i in range(1, 11):
        kmean = KMeans(n_clusters=i, init_plus=True, max_iter=100, seed=42)
        kmean.fit(x_mall)
        inertia.append(kmean.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(range(1, 11))
    plt.grid()
    plt.show()

    # Finding the optimal number of clusters using the Elbow method (KMeans simple initialization)
    inertia = []
    for i in range(1, 11):
        kmean = KMeans(n_clusters=i, init_plus=False, max_iter=100, seed=42)
        kmean.fit(x_mall)
        inertia.append(kmean.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(range(1, 11))
    plt.grid()
    plt.show()