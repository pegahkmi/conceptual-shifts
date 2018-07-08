"""use elbow methos to determine the numbber of clusters for each category"""
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pylab as plt


feat = np.load('features/features_bridge.npy')


from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans



def elbow(df, n):
	KMeansVar = [KMeans(n_clusters = k).fit(df) for k in range(1,n)]
	centroids = [X.cluster_centers_ for X in KMeansVar]
	k_euclid = [cdist(df, cent, 'euclidean') for cent in centroids]
	dist = [np.min(ke, axis=1) for ke in k_euclid]
	wcss = [sum(d**2) for d in dist]
	tss = sum(pdist(df)**2)/df.shape[0]
	bss = tss - wcss

	plt.grid(True)
	plt.ylabel('percentage of variance explained')
	plt.xlabel('number of clusters')
	plt.title('Elbow for Kmeans clustering')
	plt.plot(bss/tss*100, "b*-")
	plt.show()


elbow(feat[:2000,:], 100)


