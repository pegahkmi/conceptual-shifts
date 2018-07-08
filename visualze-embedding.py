
import numpy as np
from time import time
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, decomposition, ensemble,
                     discriminant_analysis, random_projection)

n_neighbors = 30

X = np.load('features_nose.npy')
n_samples, n_features = X.shape
 
labels = np.load('kmeans-results/kmeans_nose.npz')
Y = labels['arr_0'][0]
Y = Y[::3]


#load images
def load_image(prefix, i):
    kernel = np.ones((20,20),np.uint8)
    images = ndimage.imread("{}{}.png".format(prefix,3*i+1))
    return cv2.erode(images,kernel,iterations = 1)


# Scale and visualize the embedding vectors
embedding = np.loadtxt('nose-output.txt')
embedding = embedding[1:]
print(embedding.shape)

x_min, x_max = np.min(embedding, 0), np.max(embedding, 0)
embedding = (embedding - x_min) / (x_max - x_min)

prefix = "nose/drawing"

plt.figure()
ax = plt.subplot(111)
title ='LargeVis embedding of the sketches'

N = M = 0
all_data = {}
for i, line in enumerate(open('nose-output.txt')):
    vec = line.strip().split(' ')
    if i == 0:
        N = int(vec[0])
        M = int(vec[1])
    elif i <= N:
        all_data.setdefault(Y[i-1], []).append((float(vec[-2]), float(vec[-1])))

colors = plt.cm.rainbow(np.linspace(0, 1, max(Y)+1))

for color, ll in zip(colors, sorted(all_data.keys())):
    x = np.array([t[0] for t in all_data[ll]])
    y = np.array([t[1] for t in all_data[ll]])
    x = (x-x_min[0]) / (x_max[0] - x_min[0])
    y = (y - x_min[1]) / (x_max[1] - x_min[1])
    plt.plot(x, y, '.', color = color, markersize = 1)


# only print thumbnails with matplotlib > 1.0
shown_images = np.array([[1., 1.]])  # just something big
for i in range(embedding.shape[0]):
    dist = np.sum((embedding[i] - shown_images) ** 2, 1)
    if i >= 1837/3:
        continue
    if np.min(dist) < 4e-3:
        # don't show points that are too close
        continue
    shown_images = np.r_[shown_images, [embedding[i]]]
    print(i)
    print(embedding[i])
    imagebox = offsetbox.AnnotationBbox(
        offsetbox.OffsetImage(load_image(prefix,i), zoom = 0.02, cmap=plt.cm.gray_r),
        embedding[i])
    ax.add_artist(imagebox)
plt.xticks([]), plt.yticks([])
if title is not None:
    plt.title(title)

plt.show()
plt.savefig('plot.png', dpi = 500)



