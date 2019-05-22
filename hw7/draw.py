import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

x_tsne = np.load("x_tsne.npy")
kmeans_fit = KMeans(n_clusters = 2).fit(x_tsne)
cluster_labels = kmeans_fit.labels_


face = []
other = []
for i, c in enumerate(cluster_labels):
	if c == 1:
		face.append(i)
	else:
		other.append(i)

plt.scatter(x_tsne[face, 0], x_tsne[face, 1], c="g")
plt.scatter(x_tsne[other, 0], x_tsne[other, 1], c="y")
#plt.axis('tight')

plt.show()