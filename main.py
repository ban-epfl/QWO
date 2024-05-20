import numpy as np 
import time 
import sys
import pickle

from QVP import QVP
from generateData import * 
from metrics import skf1, pshd
from config import CONFIG


if(CONFIG.path_to_data == ""):
    G = generate_graph_erdos(CONFIG.n, CONFIG.average_deg)
    D, B, N = generate_data(G, CONFIG.number_of_samples, CONFIG.noise_dist) 

else: 
    with open(CONFIG.path_to_data, "rb") as f: 
        D = np.array(pickle.load(f))

print("#######  Running the experiment on " + str(D.shape[0]) + " data points of " + str(D.shape[1]) + " variables, using "
       + CONFIG.search_method + " search method  #######")

rtime = time.perf_counter()
B_pred = QVP(D, search_method=CONFIG.search_method, param=CONFIG.search_params[CONFIG.search_method])
rtime = time.perf_counter() - rtime
sys.stdout.write("\nThe algorithm completed in: %.2fs \n" % rtime) 

if(CONFIG.path_to_data == ""):
    print("SKF1 score of the output = ", skf1(B, B_pred))
    print("PSHD of the output =", pshd(B, B_pred))
else: 
    with open("outputs/" + CONFIG.output_file_name, "wb") as f: 
        pickle.dump(B_pred, f)

