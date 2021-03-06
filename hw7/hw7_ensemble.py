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
	train_x, test_x = load_data()
	model_path = './models/model_best.h5'
	output_path = './result/ans.csv'
	# model = load_model(model_path)
	# for i,layer in enumerate(model.layers):
	# 	print (i,layer)
	# for i in range(9):
	# 	model.pop()

	# print ("Doing t-SNE...")
	# encoded_x = preprocess(train_x, model)
	# tsne = TSNE(n_components=2, perplexity=50)
	# x_tsne = tsne.fit_transform(encoded_x)
	# np.save("x_tsne.npy", x_tsne)

	x_tsne_1 = np.load("tsne_data/x_tsne_1.npy")
	x_tsne_2 = np.load("tsne_data/x_tsne_2.npy")
	x_tsne_3 = np.load("tsne_data/x_tsne_3.npy")

	print ("Doing KMeans...")
	kmeans_fit_1 = KMeans(n_clusters = 2).fit(x_tsne_1)
	cluster_labels_1 = kmeans_fit_1.labels_
	kmeans_fit_2 = KMeans(n_clusters = 2).fit(x_tsne_2)
	cluster_labels_2 = kmeans_fit_2.labels_
	kmeans_fit_3 = KMeans(n_clusters = 2).fit(x_tsne_3)
	cluster_labels_3 = kmeans_fit_3.labels_

	cluster_labels = np.rint(((cluster_labels_1+cluster_labels_2+cluster_labels_3)/3.))

	# pca = PCA(n_components=2, svd_solver='full')

	cluster1 = cluster_labels[test_x[:,0]-1]
	cluster2 = cluster_labels[test_x[:,1]-1]
	ans_list = np.equal(cluster1,cluster2).astype(np.int)

	output_file = pd.DataFrame(ans_list, columns=['label'])
	new_ids=output_file.index.set_names("id")
	output_file.index=new_ids
	output_file.to_csv(output_path, index=True, header=True)
	print ("Generate ans.csv!")




