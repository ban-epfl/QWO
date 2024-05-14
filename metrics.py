import numpy as np
import causaldag as cd
import pandas as pd
from sklearn.metrics import f1_score


def skf1(B, B_pred): 
    """
    Compute the F1 score of the skeleton of the graph
    """
    skb = np.array(supp(np.abs(B) + np.abs(B.T))).flatten()
    skb_pred = np.array(supp(np.abs(B_pred) + np.abs(B_pred.T))).flatten()
    return f1_score(skb, skb_pred) 

def pshd(B, B_pred):
  """
  Compute the SHD between CPDAGs of the two graphs
  """
  cpdag = make_dag(np.array(supp(B))).cpdag()
  cpdag_pred = make_dag(np.array(supp(B_pred))).cpdag()
  print(make_dag(np.array(supp(B))))
  print(make_dag(np.array(supp(B_pred))))
  print(make_dag(np.array(supp(B))).cpdag())
  print(make_dag(np.array(supp(B_pred))).cpdag())
  print(cpdag.shd(cpdag_pred))
  return (cpdag.shd(cpdag_pred) / len(B))

def make_dag(arr):
  """
  Convert a matrix to a DAG object
  """
  dframe = pd.DataFrame(arr)
  return cd.DAG.from_dataframe(dframe)

def supp(A):
  """
  Compute the support of a matrix
  """ 
  return [[1 if abs(x)>1e-3 else 0 for x in row] for row in A]