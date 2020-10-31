To Run:

myObj = scQcut()
clust, net, k, clusts, dqn, range1 = myObj.getAutoAsymGeomComm(data, metric)

Inputs:
% data: normalized single cell data (rows are cells and columns are genes)
% metric: distance metric to calculate knn (default value is correlation)



Outputs:
% clust: Cell types obtained from scQcut 
% net: optimal knn network
% k: optimal number of neighbors
% clusts: clustering result for different values of k (range)
% dqn: the difference of modularity of real network and random network
% range1: different values for nearest neighbor (k)