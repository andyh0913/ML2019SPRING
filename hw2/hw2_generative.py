import numpy as np
import sys

# raw data
def preprocess(path_x, path_y):
	x_train  = np.genfromtxt(path_x, delimiter=',')[1:]
	y_train = np.genfromtxt(path_y, delimiter=',')[1:]
	# x_train.shape = (32561, 106)
	# y_train.shape = (32561, )
	return x_train, y_train

def calc():
	global x_train, y_train

	nCols = x_train.shape[1]
	nRows = y_train.shape[0]

	num = np.zeros(2)
	mu = np.zeros([2,nCols])
	sigma = np.zeros([2,nCols,nCols])

	num[1] = y_train.sum()
	num[0] = nRows - num[1]

	mu[0] = x_train[np.where(y_train==0)].mean(axis=0)
	mu[1] = x_train[np.where(y_train==1)].mean(axis=0)
	
	sigma[0] = np.cov(x_train[np.where(y_train==0)],rowvar=False)
	sigma[1] = np.cov(x_train[np.where(y_train==1)],rowvar=False)

	sigma = float(num[0])/nRows*sigma[0] + float(num[1])/nRows*sigma[1]

	np.save('sigma.npy', sigma)
	np.save('mu.npy', mu)
	np.save('num.npy', num)

if __name__ == "__main__":
	path_x = 'data/X_train'
	path_y = 'data/Y_train'
	# data = preprocess(path)
	# print (data.shape)
	x_train, y_train = preprocess(path_x, path_y)
	calc()

