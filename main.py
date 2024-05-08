import numpy as np 

from QVP import QVP_grasp, QVP_Hill_Climbing
from generateData import * 
from preprocess import preprocess


n = 50
number_of_samples = 5000
average_deg = 1
erdos_p = 2/(n-1) * average_deg
THRESHOLD = find_Fisher_threshold(number_of_samples, n)
depth = 3
G = generate_graph_erdos(n, erdos_p)
D, B, N = generate_data_gaussian(G, number_of_samples)
V, theta = preprocess(D)

B_pred = QVP_grasp(V, depth=depth) 
B_pred2 = QVP_Hill_Climbing()