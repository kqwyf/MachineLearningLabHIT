import numpy as np

def extract_pca(data, k):
    s = data.T @ data
    e, v = np.linalg.eig(s)
    indices = np.argsort(e)[-1:-k-1:-1]
    pca = v[:,indices]
    data = data @ pca
    return pca.T, data

def recover_from_pca(pca, data):
    return data @ pca
