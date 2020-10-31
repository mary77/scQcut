# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist,squareform
from math import inf, pow
from sklearn.metrics import pairwise_distances
import scipy.sparse.linalg as sla
from numpy.matlib import repmat
from sklearn.cluster import SpectralClustering
import warnings
from collections import defaultdict
from sklearn.cluster import KMeans
from numpy import linalg as la
from utils import getClusts,getMembership,Q
#import Utils.utils as utils
import copy
from sklearn.metrics.pairwise import euclidean_distances

class scQcut:
    
    def clusterInfo(self, clus, b):
        n = b.shape[0]
        m = len(clus)
        ft = sparse.lil_matrix((n,m))
        ct = sparse.lil_matrix((m,m))        
        edgeCounts = []
        label = np.zeros(n)
        for i in range(m):
            label[clus[i]] = i
            for j in range(m):
                x = b[clus[i]][:,clus[j]]
                ft[clus[i], j] = np.sum(x,1).ravel() # num of friends in community j
                ct[i,j] = np.sum(np.sum(x)) # number of edges within community or connecting communities
            edgeCounts.append(np.sum(np.sum(b[clus[i],:]))) 
        return ft, label, edgeCounts, ct

    
    def gcuts2(self, A, groupNum):
        ## using sklearn spectral clustering
        #nClusts = np.max(groupNum)
        clustering = SpectralClustering(assign_labels="discretize", random_state=0).fit(A)
        clustsLabels = clustering.labels_
        clusts = getClusts(clustsLabels)
        return clusts
    
        
    def QRefineCommunity2(self, clus, b, nIter):
        '''
        refine communities by move nodes around or merge communities
        '''
        q = []
        n = b.shape[0]
        m = len(clus)
        degree = np.sum(b,1)
        N = np.sum(degree)
        i = 0
        q.append(Q(clus, b))
        ft, label, edgeCounts, ct = self.clusterInfo(clus, b)
        ft = ft.todense()
        edgeCounts = np.array(edgeCounts,ndmin=2,dtype=np.int64)
        label = label.astype(int)
        D = sparse.diags(degree,0, format = 'csr')
        oldClus = []
        while i < nIter:
            oldClus.append(clus)
            fstarVec = [ft[i,label[i]] for i in range(ft.shape[0])]
            fstar = repmat(np.array(fstarVec),m,1)
            fstar = fstar.T
    
            edgeCountStar = np.array([edgeCounts[0,l] for l in label])
            imp = ft - fstar
            imp = imp + D * (repmat(edgeCountStar - degree,m,1).T - repmat(edgeCounts, n,1)) / N
            imp = 2 * imp / N
            
            x = np.max(np.max(imp))
            
            imp2 = (N * ct) - edgeCounts.T.dot(edgeCounts)
            imp2 = 2 * imp2 / pow(N,2)
            imp2 = imp2 - np.diag(np.diag(imp2))
            y = np.max(np.max(imp2))
            
            if x <= 0 and y <= 0:
                break # no improvement possible
            else:
                # move a single node instead of merge two communities
                if x > y:
                    I , J = np.where(imp == x)
                    node = I[0]
                    comm = J[0]
                    clus[label[node]] = list(set(clus[label[node]]) - set([node]))
                    clus[comm].append(node)
                    # update edgeCounts
                    edgeCounts[0,label[node]] = edgeCounts[0,label[node]] - degree[node]
                    edgeCounts[0,comm] = edgeCounts[0,comm] + degree[node]
                    # update ft
                    friends = sparse.find(b[node])[1]
                    edgeWeights = np.matrix(b[node, friends])
                    temp = ft[friends, label[node]] - edgeWeights.T
                    ft[friends, label[node]] = temp.ravel()
                    temp = ft[friends, comm] + edgeWeights.T
                    ft[friends, comm] =  temp.ravel()
                   
                    #update ct
                    affectedComm = list(set([label[node]]).union(set(sparse.find(ft[node])[1])))
                    for c in affectedComm:
                        ct[c,:] = np.sum(ft[clus[c],:],0)
                   
                    #update label
                    label[node] = comm
                    i +=1
                    q.append(q[i-1] + x)
                else:
                    I , J = np.where(imp2 == y) 
                    comm1 = I[0]
                    comm2 = J[0]
                    filter1 = list(set(range(len(clus))) - set([comm2]))
                    clus[comm1] = clus[comm1] + clus[comm2]
                    clus = [clus[i] for i in filter1]
                    ft[:,comm1] = ft[:,comm1] + ft[:,comm2]
                    ft = ft[:,filter1]
                    label = getMembership(clus)
                    
                    edgeCounts[0,comm1] = edgeCounts[0,comm1] + edgeCounts[0,comm2]
                    edgeCounts = edgeCounts[:,filter1]
                    
                    ct[:, comm1] = ct[:, comm1] + ct[:, comm2]
                    ct[comm1, :] = ct[comm1, :] + ct[comm2, :] 
                    ct = ct[filter1][:,filter1]
                    
                    m -= 1
                    i += 1
                    q.append(q[i-1] + y)
                    
        newClus = []
        newClus = [clus[i] for i in range(len(clus)) if len(clus[i]) > 0]
        q = q[-1]
        return newClus, q
    
    def gcutsRuan(self, A, groupNum):
        '''
        %% [clusts,distortion]=gcut(A,nClusts)
        %%
        %% Graph partitioning using spectral clustering. 
        %% Input: 
        %%   A = Affinity matrix
        %%   nClusts = number of clusters 
        %% 
        %% Output:
        %%   clusts = a cell array with indices for each cluster
        %%   distortion = the distortion of the final clustering 
        %%
        %% Algorithm steps:
        %% 1. Obtain Laplacian of the affinity matrix
        %% 2. Compute eigenvectors of Laplacian
        %% 3. Normalize the rows of the eigenvectors
        %% 4. Kmeans on the rows of the normalized eigenvectors
        %%
        %% Original code by Yair Weiss
        %% Modified and Updated by Lihi Zelnik-Manor 
        %%
        '''
        
        clusts = []
        groupNum = sorted(np.array(groupNum)) # sort the group numbers
        groupNum = [g for g in groupNum if g not in [1]] # do not allow for 1 group
        ## compute the Laplacian
        
        useSparse = sparse.issparse(A)
        dd = np.sum(A,0) + np.finfo(float).eps
        if useSparse:
            D = sparse.diags(dd, 0, format = 'csr')
        else:
            D = np.diag(dd)
        L = D - A
        nClusts = np.max(groupNum)
        L = sparse.csr_matrix(L)
        
        if nClusts == L.shape[0]:
            if nClusts==2:
                clusts = [[[0]],[[1]]]
            else:
                clusts = [[[0,1,2]]]
            return clusts
        # eigsh for symetric and eigs for non-symetric
        ss, V = sla.eigsh(L , k = nClusts, M = D , which = 'SM', tol=1E-2)
    
        I = np.argsort(ss)
        V = V[:,I[:nClusts]] 
        
        # this code solves the problem of having less than nClusts meaningful eigenvectors
        i,j = np.where(np.isnan(V))
        # index of non NaN eigvectors 
        xx = [a for a in range(nClusts) if a not in np.unique(j)]

        isreal1 = np.sum(np.sum(np.isreal(V)))
        if isreal1 == 0:
            i,j = np.where(np.imag(V))
            xx = [a for a in range(nClusts) if a not in np.unique(j)]
            warnings.warn('something is wrong')
            
        if len(xx) < 2  :
            clusts = [[range(A.shape[0])]]
            #distortion = 0
            warnings.warn('something is wrong')
            return(-1)
            
        elif len(xx) < nClusts:
            V = V[:,xx]
            groupNum = groupNum[np.where(groupNum <= len(xx))]
            warnings.warn('something is wrong')
               
        for g in groupNum:
            Vcopy = V.copy()
            Vcurr = Vcopy[:,:g]
            ## normalize rows of V
            for r in range(Vcurr.shape[0]):
                Vcurr[r,:] = Vcurr[r,:] / (la.norm(Vcurr[r,:] + np.finfo(float).eps))
            c = self.myfkmeans(Vcurr, g, 10)
            clusts.append(c)   
        return clusts
    
    def Qcut2(self, cluster, A , m, alpha, beta=0.3):
        '''
        % [cluster, q] = Qcut_2 (cluster, A, m, alpha, beta) finds a graph partitioning that (approximately) optimizes Q.
        % input:
        % cluster - a cell array, where each element is a row vector containing the node IDs in a community.  Initial value of cluster should be {1:length(A)}. 
        % A - an adjacency matrix. It needs to be symmetric.
        % m - a vector of small values, for example 2, or 2:3, or 2:4. At each step Qcut looks for the best k-way partition, where k is a value from m. I usually use m = 2:3.
        % alpha - minimum improvement of Q to accept a partition. I use 0 or eps. May use negative value (e.g., -0.05) if you want the graph to be partitioned into more clusters. 
        % beta - used when alpha < 0. the minimum local Q value required to accept a partition. suggested value is 0.25 or higher. 
        % output:
        % cluster: the community structure found
        % q: the Q value of the community structures. -1 < Q < 1. The higher the better. Should be greater than 0.3 for most real networks and close to 0 for random networks.
        '''
        n = len(cluster)
        q = []
        q.append(Q(cluster, A))
        i = 0
        while i < n:           
            # if a cluster contains less than 2 nodes, continue
            if len(cluster[i]) < 2:
                i += 1
                continue
            a = A[cluster[i]][:,cluster[i]]
            #if a.shape[0] in m:
                #clusts = [[[c] for c in range(a.shape[0])]]  
            if a.shape[0] < np.max(m):
                clusts = self.gcutsRuan(a, range(2,a.shape[0]+1))
            else:
                clusts = self.gcutsRuan(a, m)
            oldQ = Q([cluster[i]], A)
            # del qs, qsl
            qs  = []
            qsl = []
            for j in range(len(clusts)):
                qsl.append(Q(clusts[j] , a))
                for k in range(len(clusts[j])):
                    clusts[j][k] = [cluster[i][x] for x in clusts[j][k]]
                # global Q value for the submatrix
                qs.append(Q(clusts[j], A) - oldQ)
            Y = np.max(qs)
            I = np.argmax(qs)
            if Y < np.finfo(float).eps:
                qsl = np.array(qsl) * (np.array(qs) >= alpha)
                Y = np.max(qsl)
                I = np.argmax(qsl)
            # if returned only a single cluster, continue
            if qs[I] < alpha  or (qs[I] < np.finfo(float).eps and qsl[I] < beta) or len(clusts[I]) < 2:
                i += 1
                continue
            else:
                subClusts = clusts[I]
                if i > 0:
                    newClus = cluster[:i]
                else:
                    newClus = []
                newClus = newClus + subClusts
                if i < (len(cluster) - 1):
                    newClus = newClus + cluster[i+1:]
                cluster = newClus
                n = len(cluster)
        q = Q(cluster, A)
        return cluster, q
    
    def QcutPlus(self, A , cluster= None):
        
        '''
        % [cluster, q] = QcutPlus (A) finds a graph partitioning that (approximately) optimizes Q, by iteratively executing Qcut and QRefineCommunity.
        % input:
        % A - an adjacency matrix. It needs to be symmetric.
        % output:
        % cluster - the community structure stored in a cell array. 
        % Each element is a row vector containing the node IDs in a community. 
        % If the network is small (say, < 2000 vertices), you can view it with showClusters(cluster, A)
        % q: the Q value of the community structures.
        '''
        q = []
        if cluster == None:
            cluster = [list(range(A.shape[0]))]
            
        kc , qV = self.Qcut2(cluster , A, range(2,4), 0)
        q.append(qV)
        if len(kc) < 2:
            cluster = kc
            q = q[-1]
            return cluster,q
        z=0
        while 1:
            skc, qv = self.QRefineCommunity2(copy.deepcopy(kc), A, 20000)
            q.append(qv)
            if q[-1] - q[-2] < np.finfo(float).eps:
                break
            kc, qV = self.Qcut2(skc, A, range(2,4), 0)
            q.append(qV)
            
            if len(skc) == len(kc):
                break
            z=z+1
        cluster = skc
        q = q[-1]
        return cluster,q
    
    def myfkmeans(self, data, nClusts, nIter):
        bestQ = np.inf
        bestC = 0
        for i in range(nIter):
            kmeans = KMeans(n_clusters =nClusts, n_init=1).fit(data)
            centers = kmeans.cluster_centers_
            minCenter = kmeans.labels_
            uDist = self.calcDist(data,centers[minCenter,:])
            quality = np.mean(uDist)
            if quality < bestQ:
                bestC = minCenter
                bestQ = quality
        clusts = []
        for i in range(nClusts):
            I = np.where(bestC == i)
            clusts.append(list(I[0]))
        return clusts
    
    
    def calcDist(self, data, center):
        '''
        %  input: vector of data points, single center or multiple centers
        % output: vector of distances
        '''
        n, dim = data.shape
        n2, dim2 = center.shape        
        if n2 == 1:
            distances = np.sum(data ** 2, 1) - (2 * data * center.T) + (center * center.T)
        elif n2 == n:
            distances = np.sum((data - center)** 2,1)
        else:
            print('wrong number of centers')           
        distances = np.sqrt(distances)     
        return distances
    
    
    def indComp(self, net):
        net = csr_matrix(net)
        nComps , labels = connected_components(net, directed = False, return_labels = True) 
        ic = defaultdict(list)
        for i, x in enumerate(labels):
            ic[x].append(i)
        ic = list(ic.values())
        return ic
    
    
    def netRewire(self, net, nIter = None):
        if nIter == None:
            nIter = 100       
        I, J, value = sparse.find(net)
        keep = np.where(I < J)[0] 
        m = len(keep)
        edges =np.vstack((I[keep], J[keep]))
        ind = np.random.uniform(0,1,m) < 0.5
        ee = edges[:,ind]
        ee = ee[[1,0],:]
        edges[:,ind] = ee
        
        newNet = net.copy()
        isSparse = sparse.issparse(newNet)
        if isSparse:
            newNet = newNet.todense()
        for i in range(nIter):
            np.random.shuffle(edges.T)
            
            for j in range(0,m-1,2):
                sEdge = edges[:,j:j+2]
                
                # check for self edge
                if (sEdge[0,0] == sEdge[0,1]) | (sEdge[0,0] == sEdge[1,1]) | (sEdge[0,1] == sEdge[1,0]) | (sEdge[1,0] == sEdge[1,1]):
                    continue
                if newNet[sEdge[0,0],
                          sEdge[0,1]] | newNet[sEdge[1,0],sEdge[1,1]]:
                    continue
                # remove old edges
                newNet[sEdge[0,0], sEdge[1,0]] = 0
                newNet[sEdge[0,1], sEdge[1,1]] = 0
                newNet[sEdge[1,0], sEdge[0,0]] = 0
                newNet[sEdge[1,1], sEdge[0,1]] = 0
                
                # add new edges
                newNet[sEdge[0,0], sEdge[0,1]] = 1
                newNet[sEdge[1,0], sEdge[1,1]] = 1
                newNet[sEdge[0,1], sEdge[0,0]] = 1
                newNet[sEdge[1,1], sEdge[1,0]] = 1
                
                edges[:,j:j+2] = sEdge.T
        if isSparse:
            newNet = sparse.csr_matrix(newNet)
        return newNet
        
    
    def getAutoAsymGeomComm(self, data, metric= None, range1 = None):
        if sparse.issparse(data):
            data = data.todense()
        nRow, nCol = data.shape
        print('number of cells:', nRow)
        print('number of genes:', nCol)
        if range1 == None:
            range1 = [int(np.floor(1.5 ** a)) for a in np.floor(np.arange(np.log2(np.log2(nRow)) , np.log2(nRow/2)/(np.log2(1.5))))]
        if metric == None:
            metric = 'correlation'
        
        if metric == 'euclidean':
            print('start calculating distance matrix ...')
            dist = euclidean_distances(data)
        else: 
            if sparse.issparse(data):
                print('start calculating distance matrix ...')
                dist = pairwise_distances(data.todese(), metric=metric)
            else:
                #dist = pairwise_distances(data, metric=metric)
                print('start calculating distance matrix ...')
                dist=pdist(data, metric)
                dist = squareform(dist)
                np.fill_diagonal(dist, inf)
        
        I = np.argsort(dist, axis = 0) + 1
        I = np.argsort(I, axis = 0) + 1
        
        R = np.minimum(I, I.T)
        
        i = 0 
        clusts = []
        rClusts = []
        qn = []
        rqn = []
        pq = []
        rpq = []
        print('start clustering ...')
        for n in range1:
            #print(n)
            net = (R <= n)
            ic = self.indComp(net)
            
            clust , q = self.QcutPlus(copy.deepcopy(net), ic)
    
            clusts.append(clust)
            qn.append(q)
            
            rNet = self.netRewire(net,10)
            ic = self.indComp(rNet)
            clust, q = self.QcutPlus(rNet, ic)
            rClusts.append(clust)
            rqn.append(q)
            
# =============================================================================
#             if np.min(np.sum(net)) == 0 :
#                 pq.append(self.perfectQ(clusts[i][:-1]))
#             else:
#                 pq.append(self.perfectQ(clusts[i]))
#                 
#                 
#             if np.min(np.sum(net)) == 0 :
#                 rpq.append(self.perfectQ(rClusts[i][:-1]))
#             else:
#                 rpq.append(self.perfectQ(rClusts[i]))
#                 
# =============================================================================
            i = i + 1
        dqn = np.array(qn) - np.array(rqn);
        I = np.argmax(dqn)
        k = range1[I]
        
        net = csr_matrix( R <= k)
        clust = clusts[I]
        return clust, net, k, clusts, dqn, range1

               
    def perfectQ(self,clusts):
        n = 0
        for i in range(len(clusts)):
            n = n+ len(clusts[i][0])
        a = 0
        b = 0
        for i in range(len(clusts)):
            alpha = len(clusts[i][0])/ n
            a = a + alpha ** 4
            b = b + alpha ** 2
        qq = 1 - a / b**2
        return qq
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    