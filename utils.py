import numpy as np
import pandas as pd
import zipfile
def mean(X): # np.mean(X, axis = 0)
  # Your code here
  return np.mean(X, axis = 0, keepdims=True)
def std(X): # np.std(X, axis = 0)
  # Your code here
  return np.std(X, axis = 0, ddof = 1,keepdims=True)
def Standardize_data(X):
  # Your code here
  return (X - mean(X))/std(X)
def covariance(X):
  ## Your code here
  return (1/(X.shape[0]-1))*(X.T@X)
def datas():
  with zipfile.ZipFile("archive.zip", 'r') as unzipfile:
    # Extract the contents of the zip file
    with unzipfile.open("Mall_Customers.csv") as file:
        data = pd.read_csv(file)
  return data