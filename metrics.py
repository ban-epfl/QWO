import numpy as np
import causaldag as cd
import pandas as pd
from sklearn.metrics import f1_score


def skf1(B, B_pred): 
    skb = np.array(supp(np.abs(B) + np.abs(B.T))).flatten()
    skb_pred = np.array(supp(np.abs(B_pred) + np.abs(B_pred.T))).flatten()
    return f1_score(skb, skb_pred) 

def pshd(B, B_pred):
  cpdag = make_dag(B).cpdag()
  cpdag_pred = make_dag(B_pred).cpdag()
  return cpdag.shd(cpdag_pred) / len(B)

def make_dag(arr):
  dframe = pd.DataFrame(arr)
  return cd.DAG.from_dataframe(dframe)

def supp(A):
  return [[1 if abs(x)>1e-3 else 0 for x in row] for row in A]