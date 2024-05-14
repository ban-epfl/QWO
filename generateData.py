import numpy as np


def generate_graph_erdos(n, average_deg):
    """
    Generate a DAG with n nodes and average degree average_deg using the Erdos-Renyi model
    """
    p = 1/(n-1) * average_deg
    arr = np.triu(np.random.rand(n, n), k=1)
    G = (arr > 1-p).astype(int)
    perm = np.random.permutation(n)
    G_permuted = G[perm, :][:, perm]

    return G_permuted

def generate_data_gaussian(G, number_of_samples):
    """
    Generate matrix B and data from a DAG G with Gaussian noise
    """
    n = G.shape[1]
    N_var = 1 + np.random.rand(n)
    noise = np.random.normal(size = (number_of_samples, n)) @ np.diag(np.sqrt(N_var))
    B = G.T * ((0.5 + 1.5 * np.random.rand(n)) * ((-1)**(np.random.rand(n)>0.5)))
    D = noise @ np.linalg.pinv(np.eye(n)-B.T)
    return D, B, N_var

def generate_data_exp(G, number_of_samples):
    """
    enerate matrix B and data from a DAG G with exponential noise
    """
    n = G.shape[1]
    N_var = 1 + np.random.rand(n)
    noise = np.random.exponential(size = (number_of_samples, n)) @ np.diag(np.sqrt(N_var))
    B = G.T * ((0.5 + 1.5 * np.random.rand(n)) * ((-1)**(np.random.rand(n)>0.5)))
    D = noise @ np.linalg.pinv(np.eye(n)-B.T)
    return D, B, N_var

def generate_data_gumbel(G, number_of_samples):
    """
    Generate matrix B and data from a DAG G with Gumbel noise
    """
    n = G.shape[1]
    N_var = 1 + np.random.rand(n)
    noise = np.random.gumbel(size = (number_of_samples, n)) @ np.diag(np.sqrt(N_var))
    B = G.T * ((0.5 + 1.5 * np.random.rand(n)) * ((-1)**(np.random.rand(n)>0.5)))
    D = noise @ np.linalg.pinv(np.eye(n)-B.T)
    return D, B, N_var