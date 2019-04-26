import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import load_model
import sys
import os
import csv


model = load_model('model.h5')
indexes = [5286, 27925, 5173, 24476, 10383, 25580, 28364]

def preprocess(path, start, end):
	x_path = "data/x_train.npy"
	y_path = "data/y_train.npy"
	if (os.path.isfile(x_path) and os.path.isfile(y_path)):
		x_train = np.load(x_path)
		y_train = np.load(y_path)
	else:
		x_train = []
		y_train = []
		with open(path, newline='') as csvfile:
			next(csvfile)
			rows = csv.reader(csvfile)
			for row in rows:
				y_train.append(int(row[0]))
				x_train.append(np.array( row[1].split() ).astype(np.float))

		x_train = np.reshape(np.array(x_train),[-1,48,48,1])
		y_train = np.array(y_train)

		np.save('data/x_train.npy', x_train)
		np.save('data/y_train.npy', y_train)

	return x_train[start:end+1], y_train[start:end+1]


def saliency_map(x, labels):
	sess = K.get_session()
	output = K.max(model.layers[22].output, axis = 1)
	# print (output)
	grads = K.gradients(output, model.input)
	# print (grads)
	w = K.abs(grads)[0]
	# print (w)
	out, grad, w = sess.run([output, grads, w], feed_dict={model.input: x})
	print (out)
	print (grad)
	print (w)
	w = w / np.amax(w,axis=(1,2,3),keepdims=True)
	for i in range(w.shape[0]):
		z = np.squeeze(w[i], axis=-1)
		fig, ax = plt.subplots(1, 2, figsize=(8,6))
		ax[0].set_title('Original Image')
		ax[0].imshow(np.squeeze(x[i], axis=-1), cmap="gray")
		ax[1].set_title('Saliency Map')
		cax = ax[1].imshow(z, cmap = 'jet')
		fig.colorbar(cax, ax = ax[1])
		print ("Show img ", i)
		plt.show()
	return

if __name__=="__main__":
	path = "data/train.csv"
	if len(sys.argv) > 1:
		start = int(sys.argv[1][1:])
		if len(sys.argv) > 2:
			end = int(sys.argv[2][1:])
		else:
			end = start
	x_train, y_train = preprocess(path, start, end)
	saliency_map(x_train, y_train)