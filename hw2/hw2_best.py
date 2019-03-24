import numpy as np
import tensorflow as tf
import sys

batch_size = 10
epochs = 1000


# checkpoint
filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


# raw data
def preprocess(path_x, path_y):
	x_train  = np.genfromtxt(path_x, delimiter=',')[1:]
	y_train = np.genfromtxt(path_y, delimiter=',')[1:]
	# x_train.shape = (32561, 106)
	# y_train.shape = (32561, )
	return x_train, y_train

def train(batch_size, epochs):
	global x_train
	global y_train

	train_history = model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True, callbacks=[checkpoint])
		

if __name__ == "__main__":
	path_x = 'data/X_train'
	path_y = 'data/Y_train'
	# data = preprocess(path)
	# print (data.shape)
	x_train, y_train = preprocess(path_x, path_y)
	train(batch_size, epochs)