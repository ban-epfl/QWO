import numpy as np 
import time 
import sys


from QVP import QVP
from generateData import * 
from preprocess import preprocess 
from metrics import skf1, pshd


n = 50
number_of_samples = 5000
average_deg = 2
erdos_p = 2/(n-1) * average_deg
depth = 3
G = generate_graph_erdos(n, erdos_p)
D, B, N = generate_data_gaussian(G, number_of_samples)
V = preprocess(D)

rtime = time.perf_counter()
B_pred = QVP(D, search_method="grasp", param=depth)
rtime = time.perf_counter() - rtime
sys.stdout.write("\nThe algorithm completed in: %.2fs \n" % rtime) 
print("SKF1 score of the output = ", skf1(B, B_pred))
print("PSHD of the output = ", pshd(B, B_pred))


#B_pred2 = QVP(V, search_method="HC", param=5)