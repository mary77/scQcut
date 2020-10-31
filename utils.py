# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import spdiags
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from math import pow



def getMembership(clus):
    vv  = [val for sublist in clus for val in sublist]
    membership = np.zeros(len(vv) , dtype = np.int8)
    for i in range(len(clus)):
        membership[clus[i]] = i
    return membership


def getClusts(membership):
    clusts = [np.where(membership == i) for i in np.unique(membership)]
    return clusts
    

def Q(clusts, A):
    q = 0
    d = np.sum(np.sum(A), dtype=np.int64)
    qArray = [ d * np.sum(np.sum(A[I][:,I])) - pow(np.sum(np.sum(A[I,:])) , 2) for I in clusts if len(I) > 0]
    qArray = np.array(qArray, dtype=np.float64)
    q = np.sum(qArray, dtype=np.int64)
    q = q / (d * d + np.finfo(float).eps)
    return q


def simulatedBlock(n, m, p1, p2, randomize = 1):
    '''
    % [b, clus] = simulatedBlocks(n, m, p1, p2, randomize)
    % n: number of nodes
    % m: number of clusters
    % p1: probability for two nodes in the same community to form an edge
    % p2: probability for two nodes in different communities to form an edge
    %     randomize: whether the nodes in the same community are placed together
    %     (default = 1). when 0 is chosen, you can see the clustering structure
    %     from imagesc(b).
    % b: adjacency matrix.
    % clus: real clusters
    '''
    b = np.random.rand(n,n)
    s = n/m
    clus = []
    
    # background
    b = np.floor(b + p2)
    allK = [range(int(np.round(s * (i))), int(np.round(s*(i+1)))) for i in range(m)]
    for k in allK:
        b[k[0]:k[-1]+1, k[0]:k[-1]+1] = np.floor(np.random.rand(len(k), len(k)) + p1) 
        clus.append(k)
        
    b = b - np.diag(np.diag(b))
    for i in range(n):
        for j in range(i-1):
            b[i,j] = b[j,i]
        
        
    if randomize:
        y = np.random.permutation(n)
        b = b[y][:,y]
        I = np.argsort(y)
        clus = [I[c] for c in clus]
        
    b = sparse.csr_matrix(b) 
    return b, clus
        
    
def simulatedUnevenBlocks(n1, n2, r = 1):
    '''
    %% now fixed as: 1000 nodes, 1x 100 nodes, 3 x 40 nodes, 9 x    
    %% 20 nodes, 40 x 15 nodes
    '''
    a = [100] + list(np.repeat(40, 3)) + list(np.repeat(20, 9)) + list(np.repeat(15, 40))
    offDiag = 1000**2 - np.sum(np.array(a)**2)
    p2 = n2 * 1000 / offDiag
    
    b = np.random.rand(1000,1000)
    b = b < p2
    I , J  = np.where(b)
    s = np.where(I < J)
    b = sparse.lil_matrix((np.ones(len(I[s])) , (I[s] , J[s])),shape = (1000,1000))
    b = b + b.T
    
    x = 0
    clus = []
    for i in range(len(a)):
        p1 = (n1 + np.log(a[i])) / a[i]
        s = simulatedBlock(a[i] , 1, p1, 0)[0]
        clus.append( x + np.array((range(a[i]))))
        b[clus[i][0]:clus[i][-1]+1 , clus[i][0]:clus[i][-1]+1] = s
        x = x + a[i]
    if r:
        y = np.random.permutation(1000)
        b = b[y][:,y]
        I = np.argsort(y)
        clus = [I[c] for c in clus]
    return b ,clus

def simulatedUnevenBlocks2(n1, n2, r = 1):
    '''
    %% fixed as: 512 nodes, 1x 128 nodes, 2 x 64 nodes, 8 x    
    %% 32 nodes
    '''
    a = [128,64,64] + list(np.repeat(32,8))
    n=  np.sum(a)
    
    offDiag = n**2 - np.sum(np.array(a) ** 2)
    p2 = n2 * n / offDiag
    
    b = np.random.rand(n,n)
    b = b < p2
    I, J = np.where(b)
    s = np.where(I < J)
    b = sparse.csr_matrix((np.ones(len(I[s])) , (I[s] , J[s])),shape = (n,n))
    b = b + b.T
    
    x = 0
    clus = []
    for i in range(len(a)):
        p1 = (n1 + np.log2(a[i])) / a[i]
        s = simulatedBlock(a[i] , 1, p1, 0)[0]
        clus.append( x + np.array((range(a[i]))))
        b[clus[i][0]:clus[i][-1]+1 , clus[i][0]:clus[i][-1]+1] = s
        x = x + a[i]
    if r:
        y = np.random.permutation(n)
        b = b[y][:,y]
        I = np.argsort(y)
        clus = [I[c] for c in clus]
    return b ,clus
    
    
    
def showClusters(a , clus, noLine = 0):
    o = []
    for c in clus:
        o = list(o) + list(c)    
    
    l = 0.5
    n = len(o) + 0.5
    for i in clus:
        l = l + len(i)
        if ~noLine:
            plt.plot([0,n], [l,l], c= 'red', linewidth = 0.3)
            plt.plot([l,l], [0,n], c ='red', linewidth = 0.3)
    plt.imshow(a[o][:,o]) 
    
# here 1-alpha is the restart probability
def RWR(A, nSteps = 500, alpha = 0.5, p0 = None):
    A = np.array(A)
    n = A.shape[0]
    if p0 == None:
        p0 = np.eye(n)
    #W = A * spdiags(sum(A)'.^(-1), 0, n, n);
    #W = spdiags(np.power(sum(np.float64(A)) , -1).T  , 0, n, n).toarray()
    W = A.dot(spdiags(np.power(sum(np.float64(A)) , -1)[np.newaxis],0, n, n).toarray() )
    p = p0
    pl2norm = np.inf
    unchanged = 0
    for i in range(1, nSteps+1):
        if i % 100 == 0:
            print('      done rwr ' + str(i-1) )
            
        pnew = (1 - alpha) * W.dot(p) + (alpha) * p0
        l2norm = max(np.sqrt(sum((pnew - p) ** 2) ))
        p = pnew
        if l2norm < np.finfo(float).eps:
            break
        else:
            if l2norm == pl2norm:
                unchanged = unchanged +1
                if unchanged > 10:
                    break
            else:
                unchanged = 0
                pl2norm = l2norm
    return p


# construct gene-gene network by getting ED similarity matrix 
def simToNetARank(sim, k=3):
    print('start ARank ...')
    # sim is similarity matrix and k is number of neighbors
    # return A graph which two vertices are connected if one of them are in k negbor of the other one
    np.fill_diagonal(sim, 0)
    I = np.argsort(sim, axis = 0) + 1
    I2 = (np.argsort(I, axis = 0) + 1)
    net = I2 > (len(sim) - k)
    net = np.logical_or(net, net.T)
    np.fill_diagonal(net, False)  
    net = net*1
    return net

def simToNetMRank(sim , k = 3):
    print('start MRank ...'  , k )
    np.fill_diagonal(sim, 0)
    # sim is similarity matrix and k is num of neigbors
    # return A graph which two vertices are connected if both of them are in k neighbor of the  other
 
    I = np.argsort(sim, axis = 0) + 1
    I = np.argsort(I, axis = 0) + 1
    net = I >  (sim.shape[0] - k)
    net = np.logical_and(net, net.T)
    np.fill_diagonal(net, False)  
    net = net*1
    return net

def eucliSim(data):
    dist = pdist(data, 'euclidean')
    dfDist = squareform(dist)
    sim = 1 / ( 1 + dfDist)
    np.fill_diagonal(sim, 0)
    return sim

def simToNetThreshold(sim , thre):
    print('start tNet ...', thre)
    np.fill_diagonal(sim , 0)
    sim = sim >= thre
    sim = sim * 1
    return sim

def readCountMat(name , sep = ','):
    df = pd.read_table(name , sep = sep)
    genes  = df.iloc[:,0]
    df.index = genes
    df = df.drop(df.columns[0] , 1)
    print('number of genes: ', df.shape[0] , 'number of cells: ', df.shape[1])
    return df

def readLabels(fileName):
    y = pd.read_table(fileName, sep = ',')
    y , yy = y.iloc[:,1].factorize()
    return y , yy

# log2 transformation
def logTransformation(df , offset = 1):
    return np.log2(df + offset)

# all cells have same sum counts
def libSizeNorm(df):
    libSize =  np.array(df.sum(0))
    df = df / libSize * np.median(libSize)
    return df

# data preprossenig remove genes that have not been expressed 
def preProssesing(data):
    index = geneFilteringIndex(data , 0)
    return data[~index,:]

# gene filtering: filter the genes which has been expressed in less than m cell
def geneFilteringIndex(df , m):
    # m: each gene should have been expressed in at least m cells
    geneFilterIndex = np.sum(df != 0 , 1) <= m
    print(np.sum(geneFilterIndex) , ' genes have been removed')
    return geneFilterIndex

def aknn(dist , k = 3):
    print('start ARank new ...')
    # dist is distance matrix and k is number of neighbors
    # return A graph which two vertices are connected if one of them are in k negbor of the other one
    np.fill_diagonal(dist, np.inf)
    I = np.argsort(dist, axis = 0) + 1
    I2 = (np.argsort(I, axis = 0) + 1)
    net = I2 <= k
    net = np.logical_or(net, net.T)
    np.fill_diagonal(net, False)  
    net = net*1
    return net

def mknn(dist, k = 3):
    print('start MRank new ...'  , k )
    np.fill_diagonal(dist, np.inf)
    # dist is distance matrix and k is num of neigbors
    # return A graph which two vertices are connected if both of them are in k neighbor of the  other
    I = np.argsort(dist, axis = 0) + 1
    I = np.argsort(I, axis = 0) + 1
    net = I <= k
    net = np.logical_and(net, net.T)
    np.fill_diagonal(net, False)  
    net = net*1
    return net


def aknnASym(dist , k = 3):
    print('start aknnASym new ...')
    # dist is distance matrix and k is number of neighbors
    # return A graph which two vertices are connected if one of them are in k negbor of the other one
    np.fill_diagonal(dist, np.inf)
    I = np.argsort(dist, axis = 0) + 1
    I2 = (np.argsort(I, axis = 0) + 1)
    net = I2 <= k
    #net = np.logical_or(net, net.T)
    #np.fill_diagonal(net, False)  
    #net = net*1
    return net

def aknnASymIndex(dist , k = 3):
    print('start aknnASym new ...')
    # dist is distance matrix and k is number of neighbors
    # return A graph which two vertices are connected if one of them are in k negbor of the other one
    np.fill_diagonal(dist, np.inf)
    I = np.argsort(dist, axis = 0) + 1
    I2 = (np.argsort(I, axis = 0) + 1)
    I2[I2>k]= 0
    return I2


def aknnAsymWeighted(dist, k = 3):
    print('start aknnASym new ...')
    # dist is distance matrix and k is number of neighbors
    # return A graph which two vertices are connected if one of them are in k negbor of the other one
    sim = 1 / ( 1 + dist)
    np.fill_diagonal(sim, 0)
    I = np.argsort(sim, axis = 0) + 1
    I2 = (np.argsort(I, axis = 0) + 1)
    net = I2 > (len(sim) - k)
    weightedNet = net * sim
    return weightedNet   
