import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def prep_data(data):
    # Dropping the 'CustomerID' column
    data = data.drop(columns=['CustomerID'])
    # Encoding the gender column
    data_encod = pd.get_dummies(data, columns=["Gender"]) #({"Male": 1, "Female": 0})
    return data_encod

# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna(axis=1)
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.savefig('scatter_matrix.png', dpi=300)
    plt.tight_layout()
    plt.show()

