import numpy as np 
import time 
import sys


from QVP import QVP_grasp, QVP_Hill_Climbing
from generateData import * 
from preprocess import preprocess 


n = 50
number_of_samples = 5000
average_deg = 2
erdos_p = 2/(n-1) * average_deg
THRESHOLD = find_Fisher_threshold(number_of_samples, n)
depth = 3
G = generate_graph_erdos(n, erdos_p)
D, B, N = generate_data_gaussian(G, number_of_samples)
V, theta = preprocess(D)

rtime = time.perf_counter()
B_pred = QVP_grasp(V, depth=depth)
rtime = time.perf_counter() - rtime
sys.stdout.write("\nThe algorithm completed in: %.2fs \n" % rtime)


B_pred2 = QVP_Hill_Climbing(V, dist_limit=5)