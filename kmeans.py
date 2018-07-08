"""cluster each category into different sub-categories using k-means algorithm"""
import argparse
import numpy as np
from sklearn.cluster import KMeans


parser = argparse.ArgumentParser(description='apply k-means clustering for each category.')
parser.add_argument('--features_path',type=str, help='The path to the features')
parser.add_argument('--num_clusters',type=int, help='The number of clusters for the category')
args = parser.parse_args()
print('features_path {}'.format(args.features_path))
print('num_clusters {}'.format(args.num_clusters))


feat = np.load(args.features_path)

kmeans = KMeans(args.num_clusters).fit(feat) 
labels = kmeans.labels_
centers = kmeans.cluster_centers_


np.savez("kmeans_nose",[labels, centers])






