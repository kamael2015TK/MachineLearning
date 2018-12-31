import numpy as np
from matplotlib.pyplot import figure, bar, title, show, plot, xticks
from scipy.stats.kde import gaussian_kde
import sys
sys.path.append('./Tools')
#from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors

def findAbnormalities(data = []): 
    N, M = data.shape
    K = 2

    # Find the k nearest neighbors
    knn = NearestNeighbors(n_neighbors=K).fit(data)
    D, i = knn.kneighbors(data)
    density_glob = 1./(D.sum(axis=1)/K)
    avg_rel_density = density_glob/(density_glob[i[:,1:]].sum(axis=1)/K)

    # Sort the scores
    i = avg_rel_density.argsort()
    density = avg_rel_density[i]
    print(density)
    print(i)

def gausKernelDensity(data = [],width = []):
    data = np.mat(np.asarray(data))
    N,M = data.shape
    x2 = np.square(data).sum(axis=1)
    D = x2[:,[0]*N] - 2*data.dot(data.T) + x2[:,[0]*N].T

    # Evaluate densities to each observation
    Q = np.exp(-1/(2.0*width)*D)
    # do not take density generated from the data point itself into account
    Q[np.diag_indices_from(Q)]=0
    sQ = Q.sum(axis=1)
    
    density = 1/((N-1)*np.sqrt(2*np.pi*width)**M+1e-100)*sQ
    log_density = -np.log(N-1)-M/2*np.log(2*np.pi*width)+np.log(sQ)
    return np.asarray(density), np.asarray(log_density)

def getGKD(data = []):
    widths = 2.0**np.arange(-10,10)
    logP = np.zeros(np.size(widths))
    for i,w in enumerate(widths):
        f, log_f = gausKernelDensity(data, w)
        logP[i] = log_f.sum()
    ind = logP.argmax()
    width=widths[ind]
    print('Optimal estimated width is: {0}'.format(width))
    density, log_density = gausKernelDensity(data, width)
    i = (density.argsort(axis=0)).ravel()
    density = density[i]
    print(i)
    print(density.squeeze())