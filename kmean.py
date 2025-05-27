"""
K-Means clustering implementation with optional k-means++ initialization and PCA for visualization.

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import datas, Standardize_data, covariance
from numpy.linalg import eig

class KMeans:
  def __init__(self, n_clusters, init_plus = False, max_iter = 1000, seed = None):
    """
    Args:
      k: (int) The number of clusters
      init_plus: (bool) If True, use k-means++ initialization, else random initialization
      max_iter: (int) Maximum number of iterations for convergence
    """

    self.seed = np.random.seed(seed)
    self.k = n_clusters
    self.init_plus = init_plus
    self.centroids = None
    self.clusters = None
    self.max_iter = max_iter
    self.inertia_ = None
  def input_check(self, x):
    """This function checks the input data for validity.
    Arg: 
      x: (numpy array (nxd)) The dataset from which we select the centroids
    Returns: None
    Raises:
      ValueError: If the input data is not valid.
    """
    if len(x.shape) == 1:
      x = x.reshape(-1, 1)
    if self.k > len(x):
      raise ValueError("Number of clusters k cannot be greater than the number of data points.")
    if self.k <= 0:
      raise ValueError("Number of clusters k must be a positive integer.")
    if x.ndim != 2:
      raise ValueError("Input data must be a 2D array.")
    if not isinstance(x, np.ndarray):
      raise ValueError("Input data must be a numpy array.")
    if x.shape[0] == 0:
      raise ValueError("Input data cannot be empty.")
    if x.shape[0] == 1:
      raise ValueError("Input data must have more than one data point.")
    if x.shape[1] <= 1:
      raise ValueError("Input data must have more than one feature.")
    return True

  def random_init(self, x):
    """This function initialises the centroids randomly.
    Arg: 
      x: (numpy array (nxd)) The dataset from which we select the centroids
      k: (int) The number of centroids
    Returns: An array (shape kxd) of centroids.
    """
    self.input_check(x)
    if self.seed:
      self.seed
    a = np.random.choice(len(x), self.k, replace = False)
    self.centroids = x[a]
    return self


  def init_for_kmean_plus(self, x):
    """This function initialises the centroids.
    One centroid is randomly selected and the
     others are selected based on their distance to the existing centroids.
      The farest are more likely to become centroids but with control on outliers.

      Arg: 
        x: (numpy array (nxd)) The dataset from which we select the centroids
        k: (int) The number of centroids
      Returns: An array (shape kxd) of centroids. 
     """
    self.input_check(x)
    self.centroids = []
    # Choose first center randomly
    if self.seed:
      self.seed
    self.centroids.append(x[np.random.randint(len(x))])

    for _ in range(1, self.k):
      # Compute squared distances to nearest center
      dist_sq = np.array([min(np.sum((y - c)**2) for c in self.centroids) for y in x])
      probs = dist_sq / dist_sq.sum()
      cumulative_probs = probs.cumsum()
      r = np.random.rand()
      for i, p in enumerate(cumulative_probs):
        if r < p:
          self.centroids.append(x[i])
          break
    self.centroids = np.array(self.centroids)
    return self
  def make_cluster_(self, x, cent):
    """This function creates clusters based on the centroids.
    Arg:
      x: (numpy array (nxd)) The dataset from which we select the centroids
      cent: (numpy array (kxd)) The centroids
    Returns: Updates the clusters attribute with the clusters formed.
    """
    self.clusters = []
    for c in cent:
      clus = []
      for point in x:
        dist = np.min([np.sum((point-ce)**2) for ce in cent])
        if np.sum((point-c)**2) == dist:
          clus.append(point)
      clus = np.array(clus)
      self.clusters.append(clus)
    return self
  def make_centroid_(self):
    """This function computes the centroids of the clusters.
    Returns: An array (shape kxd) of centroids.
    """
    self.centroids = []
    for clus in self.clusters:
      if len(clus)>0:
        self.centroids.append(np.mean(clus, axis = 0))
    self.centroids = np.array(self.centroids)
    return self.centroids
  def convergence_(self, cent):
    """This function checks if the centroids have converged.
    Arg:
      cent: (numpy array (kxd)) The previous centroids
    Returns: True if the centroids have converged, False otherwise.
    """
    count = 0
    tol = 1e-5
    for i in range(self.k):
      if np.linalg.norm(self.centroids[i] - cent[i]) < tol:
        count += 1
    return count == self.k

  def plot_(self):
    """This function plots the clusters and centroids.
    It uses PCA to reduce the dimensionality of the data to 2D for visualization.
    Returns: None
    """
    self.cl_plot = self.clusters
    self.cent_plot = self.centroids
    cent = PCA1()
    clust = PCA1()
    if self.x.shape[1] > 2:
      cent.fit(self.cent_plot)
      self.cent_plot = cent.transform()
      for i in range(self.k):
        if len(self.cl_plot[i])>0:
          clust.fit(self.cl_plot[i])                    
          self.cl_plot[i] = clust.transform()
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plotting points for each cluster
    for i, cluster in enumerate(self.cl_plot):
        if len(cluster) > 0:
          cluster = np.array(cluster)
          ax.scatter(cluster[:, 0], cluster[:, 1], cmap='viridis', marker='o', s=50, alpha=0.6)

    # Plotting centroids
    for point in self.cent_plot:
        ax.scatter(*point, linewidth=2,  c='red', marker='X', s=200, label='Centroids')

    plt.show()

  def step(self, i):
    """This function prints the current iteration number.
    Arg:
      i: (int) The current iteration number
    Returns: None
    """
    if i+1 > 3:
      print(f"This is the {i+1}th iteration")
    elif i+1 == 2:
      print(f"This is the {i+1}nd iteration")
    elif i+1 == 3:
      print(f"This is the {i+1}rd iteration")
    else:
      print(f"This is the {i+1}st iteration")

  def fit(self, x, plot = False):
    """This function fits the KMeans model to the data.
    Args:
      x: (numpy array (nxd)) The dataset from which we select the centroids
      plot: (bool) If True, plot the clusters and centroids after each iteration
    Returns: None
    """
    self.x = x
    # Initialise centroids
    if self.init_plus:
      self.init_for_kmean_plus(x)
    else:
      self.random_init(x)
    # creating clusters and updating centroids
    i = 0
    while i < self.max_iter:
      if plot:
        self.step(i)
      cent = self.centroids.copy()
      self.make_cluster_(x, self.centroids)
      self.make_centroid_()
      if self.convergence_(cent):
        print(f"Convergence reached after {i + 1} steps!!!")
        self.number_of_iterations = i + 1
        if plot:
          self.plot_()
        break
      i += 1
  def inertia(self):
    """This function computes the inertia of the clusters.
    Returns: The inertia value, which is the sum of squared distances of samples to their closest cluster center.
    """
    if self.clusters is None or self.centroids is None:
      raise ValueError("Clusters and centroids must be computed before calculating inertia.")
    self.inertia_ = 0
    for i, cluster in enumerate(self.clusters):
      if len(cluster) > 0:
        self.inertia_ += np.sum([(cluster[j] - self.centroids[i])**2 for j in range(len(cluster))])
    return self.inertia_
  def labels(self):
    """This function returns the labels of the clusters.
    Returns: An array of labels for each data point, indicating the cluster it belongs to.
    """
    if self.clusters is None:
      raise ValueError("Clusters must be computed before getting labels.")

    # Flatten all points and assign labels
    all_points = []
    labels = []

    for cluster_idx, cluster in enumerate(self.clusters):
      for point in cluster:
          all_points.append(point)
          labels.append(cluster_idx)

    # Convert to NumPy arrays for use in analysis or plotting
    self.x = np.array(all_points)
    self.labels = np.array(labels)
    return self.labels
  

class PCA1:
  """Principal Component Analysis (PCA) implementation for dimensionality reduction.
  This class standardizes the data, computes the covariance matrix, and transforms the data
  into a lower-dimensional space using the top principal components.
  """

  def __init__(self, n_components = 2) -> None:
    self.N = n_components
    self.X = None

  def Cov_mat(self):
    """Computes the covariance matrix of the standardized data.
    Returns:
      A: Covariance matrix of the standardized data.
      eigvals: Eigenvalues of the covariance matrix.
      eigen_vectors: Eigenvectors of the covariance matrix.
    """

    A = covariance(self.X)
    eigvals, eigen_vectors = eig(A)
    self.covMat = A
    return A, eigvals, eigen_vectors

  def Standardize_data(self, X):
    """Standardizes the input data.
    Args:
      X: (numpy array) The input data to be standardized.
    Returns:
      Standardized data as a numpy array.
    """
    return Standardize_data(X)

  def transform(self):
    """Transforms the standardized data into a lower-dimensional space using PCA.   
    Returns:
      Transformed data in the lower-dimensional space.
    """
    _, Eval, Evec = self.Cov_mat()
    idx = np.array([np.abs(i) for i in Eval]).argsort()[::-1]
    Evec_sorted = Evec.T[:,idx]
    Projec = Evec_sorted[:self.N, :] # Projection matrix
    return self.X.dot(Projec.T)

  def fit(self, X):
    """Fits the PCA model to the input data.
    Args:
      X: (numpy array) The input data to be fitted.
    Returns:
      self: The PCA instance with standardized data.
    """
    X = np.array(X)
    self.X = self.Standardize_data(X)
    return self.X
