import numpy as np
from scipy.linalg import sqrtm
from scipy.linalg import inv
import copy

def preprocess(D):
  normal_theta = compute_theta(D)
  V = compute_V(normal_theta)
  return V, normal_theta

def compute_theta(D):
  cov_matrix = np.cov(D.T)
  theta = inv(cov_matrix)
  return normalize(theta)

def compute_V(theta):
  return sqrtm(theta)

def normalize(X):
  normal_X = copy.deepcopy(X)
  for i in range(len(X)):
    for j in range(len(X)):
      normal_X[i, j] = X[i,j] / np.sqrt(X[i,i] * X[j,j])
  return normal_X
