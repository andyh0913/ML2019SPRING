import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model
from sklearn.manifold import TSNE
from sklearn import cluster
import keras.backend as K

batch_size = 128

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

	return train_x


if __name__ == '__main__':
	train_x = load_data()
	model_path = './models/model_best.h5'
	model = load_model(model_path)
	# for i,layer in enumerate(model.layers):
	# 	print (i,layer)

	while(True):
		index = input("Please enter the index: ")
		index = int(index)
		if index < 0:
			print ("Exit!")
			break
		x = train_x[index:index+1]
		arr = model.predict(x)[0]
		arr = (arr*127.5+127.5).astype(np.uint8)
		x_arr = (x[0]*127.5+127.5).astype(np.uint8)
		img = Image.fromarray(arr)
		x_img = Image.fromarray(x_arr)
		x_img.show()
		img.show()




