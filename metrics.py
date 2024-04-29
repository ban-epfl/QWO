import numpy as np
import causaldag as cd
import pandas as pd


def skeleton_error(B, B_pred):
  acc = np.array(supp((B_pred + B_pred.T))) - (np.array(supp(B + B.T)))
  return (len(B)**2 - len(acc[acc == 0]))//2

def orientation_error(B, B_pred):
  cpdag = make_dag(B).cpdag()
  cpdag_pred = make_dag(B_pred).cpdag()
  return cpdag.shd(cpdag_pred)

def make_dag(arr):
  dframe = pd.DataFrame(arr)
  return cd.DAG.from_dataframe(dframe)

def cldag2dag(g):
  g[g < 0] = 0
  g[g > 0] = 1
  return g