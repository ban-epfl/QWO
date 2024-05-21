"""Configuration of an experiment."""

from easydict import EasyDict as edict

# Get the configuration dictionary whose keys can be accessed with dot.
CONFIG = edict()

# Data generation parameters
CONFIG.n = 50                                 # Number of variables
CONFIG.number_of_samples = 10000              # Number of data points
CONFIG.average_deg = 2                        # Average degree of nodes for Erdos-Renyi graph
CONFIG.noise_dist = "gaussian"                # Distribution of data noise. options = [gaussian, exp, gumbel]

# Search method 
CONFIG.search_method = "grasp"                # Options = [grasp, HC]
CONFIG.search_params = {"grasp":3, "HC": 5}   # Depth of DFS in grasp search method and Maximum distance of indices in Hill Climbing search method

# Parameters for existing data
CONFIG.path_to_data = ""                      # Path to the pickle file containing the matrix of data
CONFIG.output_file_name = "result"            # File name for saving the adjacency matrix of the predicted graph
