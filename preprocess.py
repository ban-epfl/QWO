import numpy as np
from scipy.linalg import sqrtm
from scipy.linalg import inv
import copy

def preprocess(D):
  """"
  Preprocess the data matrix D, returning the matrix V 
  """
  normal_theta = compute_theta(D)
  V = compute_V(normal_theta)
  return V

def compute_theta(D):
  """
  Compute the precision matrix of the data matrix D
  """
  cov_matrix = np.cov(D.T)
  theta = inv(cov_matrix)
  return normalize(theta)

def compute_V(theta):
  """
  Compute the matrix V from the precision matrix theta
  """
  return sqrtm(theta)

def normalize(X):
  """
  Normalize the matrix X by dividing each element by the square root of the product of the diagonal elements
  """
  normal_X = copy.deepcopy(X)
  for i in range(len(X)):
    for j in range(len(X)):
      normal_X[i, j] = X[i,j] / np.sqrt(X[i,i] * X[j,j])
  return normal_X
