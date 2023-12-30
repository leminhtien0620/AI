import numpy as np

_polyDegree = 2
_gaussSigma = 1

def myPolynomialKernel(X1, X2):
    '''
    Arguments:  
        X1 - an n1-by-d numpy array of instances
        X2 - an n2-by-d numpy array of instances
    Returns:
        An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    return (np.dot(X1, X2.T) + 1) ** _polyDegree

def myGaussianKernel(X1, X2, gaussSigma):
    '''
    Arguments:
        X1 - an n1-by-d numpy array of instances
        X2 - an n2-by-d numpy array of instances
        gaussSigma - the value of sigma in the Gaussian kernel
    Returns:
        An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    n1, d = X1.shape
    n2 = X2.shape[0]

    # Initialize the Gram matrix with zeros
    K = np.zeros((n1, n2))

    # Compute the Gaussian kernel for all pairs of instances
    for i in range(n1):
        for j in range(n2):
            squared_distance = np.sum((X1[i] - X2[j]) ** 2)
            K[i, j] = np.exp(-squared_distance / (2 * gaussSigma**2))

    return K
    '''
    Arguments:
        X1 - an n1-by-d numpy array of instances
        X2 - an n2-by-d numpy array of instances
    Returns:
        An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            diff = X1[i] - X2[j]
            K[i, j] = np.exp(-np.dot(diff, diff) / (2 * _gaussSigma**2))
    
    return K

def myCosineSimilarityKernel(X1, X2):
    '''
    Arguments:
        X1 - an n1-by-d numpy array of instances
        X2 - an n2-by-d numpy array of instances
    Returns:
        An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    norm_X1 = np.linalg.norm(X1, axis=1, keepdims=True)
    norm_X2 = np.linalg.norm(X2, axis=1, keepdims=True)
    return np.dot(X1, X2.T) / np.dot(norm_X1, norm_X2.T)
