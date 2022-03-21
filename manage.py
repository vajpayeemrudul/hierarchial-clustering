import pandas as pnd
import numpy as npy
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


P=pnd.read_csv('mrudulpython.csv')
Q=P.iloc[:].values

print(P.shape)
print(P.head())

############  dendogram plot  ##########
linked = linkage(Q, 'single')
labelList = range(0,7)
plt.figure(figsize = (10, 7))
dendrogram(linked, orientation = 'top',labels = labelList, distance_sort='descending',show_leaf_counts = True)
plt.title('Dendrogram')
plt.xlabel('Students')
plt.ylabel('Euclidean distances')
plt.show()

####### Scatter plot ##########
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage ='ward')
y_hc=hc.fit_predict(P)
plt.scatter(Q[y_hc==0, 0], Q[y_hc==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(Q[y_hc==1, 0], Q[y_hc==1, 1], s=100, c='black', label ='Cluster 2')
plt.scatter(Q[y_hc==2, 0], Q[y_hc==2, 1], s=100, c='purple', label ='Cluster 3')
plt.title('Clusters')
plt.xlabel('Student')
plt.ylabel('Distances')
plt.show()