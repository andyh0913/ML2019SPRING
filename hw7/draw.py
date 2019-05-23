import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model
# from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import keras.backend as K
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt

batch_size = 1000

def load_data(image_directory='./data/images', csv_path='./data/test_case.csv'):
	if os.path.exists("./train_x.npy"):
		print ("Loading ./train_x.npy")
		train_x = np.load("./train_x.npy")
	else:
		print ("Loading image data...")
		train_x = []
		for i in range(1,40001):
			file_path = os.path.join(image_directory, ("%06d" % i) +'.jpg')
			img = Image.open(file_path)
			arr = np.array(img)
			train_x.append(arr)
		train_x = (np.array(train_x)-127.5)/127.5
		np.save("./train_x.npy", train_x)
		
	test_data = pd.read_csv(csv_path).values[:,1:]
	print (test_data)
	return train_x, np.array(test_data)

	# img_path = os.path.join(image_directory, )

def preprocess(train_x, model):
	length = train_x.shape[0]
	iterations = length // batch_size
	encoded_x = model.predict(train_x, batch_size=batch_size)
	print (encoded_x.shape)
	return encoded_x


if __name__ == '__main__':
	data = np.load('visualization.npy')
	model_path = './models/model_best.h5'
	model = load_model(model_path)
	for i,layer in enumerate(model.layers):
		print (i,layer)
	for i in range(9):
		model.pop()

	print ("Doing t-SNE...")
	encoded_x = preprocess(data, model)
	tsne = TSNE(n_components=2, perplexity=50, n_jobs=8, verbose=1)
	x_tsne = tsne.fit_transform(encoded_x)
	np.save("x_tsne.npy", x_tsne)

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

	# face = np.arange(2500)
	# other = face + 2500

	plt.scatter(x_tsne[face, 0], x_tsne[face, 1], c="y")
	plt.scatter(x_tsne[other, 0], x_tsne[other, 1], c="g")
	#plt.axis('tight')

	plt.show()