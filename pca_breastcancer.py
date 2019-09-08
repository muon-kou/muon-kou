#%matplotlib inline
from sklearn.cluster import AffinityPropagation, KMeans, DBSCAN, SpectralClustering
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.metrics import silhouette_score

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import logm, expm
from contrastive import CPCA

sheat = pd.read_csv("TNBC10vNormal10_Counts_4.csv", sep=",",header=0, index_col=0)
sheat2 = sheat.T
print(sheat2)

sheat2 = pd.DataFrame(sheat2)

X,y = sheat2.iloc[:, :].values, np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])

foreground_data = X[:,:]
background_data = X[10:20,:]
background_data
mdl = CPCA()
pre_cluster_lables = np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
                            
projected_data = mdl.fit_transform(foreground_data, background_data, plot=True,active_labels=pre_cluster_lables)


