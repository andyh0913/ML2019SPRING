import numpy as np
import sys

batch_size = 26048 # 5652 = 471 * 12
epochs = 200000
lr = 0.01
valid_rate = 0.2
lamb = 0.95


# raw data
def preprocess(path_x, path_y):
	x_train  = np.genfromtxt(path_x, delimiter=',')[1:]
	y_train = np.genfromtxt(path_y, delimiter=',')[1:]
	# x_train.shape = (32561, 106)
	# y_train.shape = (32561, )

	# add square term (index:0, 1, 3, 4, 5)
	x_squr = np.concatenate([x_train[:,0:2], x_train[:,3:6]], axis=1) ** 2
	x_train = np.concatenate([x_train,x_squr], axis=1)
	
	# normalization
	x_mean = x_train.sum(axis=0) / x_train.shape[0]
	x_var = ((x_train - x_mean) ** 2).sum(axis=0) / x_train.shape[0]
	x_train = (x_train - x_mean)/(x_var**0.5)

	np.save('x_mean.npy', x_mean)
	np.save('x_var.npy', x_var)

	n_valid = (int)(x_train.shape[0]*valid_rate)

	return x_train[:-n_valid], y_train[:-n_valid], x_train[-n_valid:], y_train[-n_valid:]

def train(batch_size, epochs, lr):
	global x_train
	global y_train
	global x_valid
	global y_valid
	global w
	global b

	iterations = x_train.shape[0] // batch_size

	prev_grad_w = 0.000001
	prev_grad_b = 0.000001

	min_loss = 10000

	for i in range(epochs):
		assert len(x_train) == len(y_train)
		p = np.random.permutation(len(x_train))
		x_train = x_train[p]
		y_train = y_train[p]
		loss = 0
		for j in range(iterations):
			x = x_train[j*batch_size:(j+1)*batch_size]
			y = y_train[j*batch_size:(j+1)*batch_size]

			z = (x * w).sum(axis=1) + b

			y_h = 1 / ( 1 + np.exp(-z) )
			delta = y-y_h
			grad_w = -( x * np.expand_dims(delta, 1 )).sum(axis=0)
			grad_b = -delta.sum()
			
			# adadelta
			# prev_grad_w = lamb*prev_grad_w + (1-lamb)*(grad_w ** 2)
			# prev_grad_b = lamb*prev_grad_b + (1-lamb)*(grad_b ** 2)

			# adagrad
			prev_grad_w += (grad_w ** 2)
			prev_grad_b += (grad_b ** 2)
			

			w -= lr * grad_w / (prev_grad_w ** 0.5)
			b -= lr * grad_b / (prev_grad_b ** 0.5)

			loss += -( y*np.log(y_h) + (1-y)*(np.log(1-y_h+0.00000001)) ).mean()
		loss /= iterations

		# validation
		z = (x_valid * w).sum(axis=1) + b
		y_h = 1 / ( 1 + np.exp(-z) )
		val_loss = -( y_valid*np.log(y_h) + (1-y_valid)*(np.log(1-y_h+0.00000001)) ).mean()

		print ("{} epoch, loss = {}, val_loss = {}".format(i+1,loss, val_loss))
		if val_loss < min_loss:
			print ("Model saved! Val_loss = {}".format(val_loss))
			np.save('w.npy',w)
			np.save('b.npy',b)
			min_loss = val_loss


if __name__ == "__main__":
	path_x = 'data/X_train'
	path_y = 'data/Y_train'
	# data = preprocess(path)
	# print (data.shape)
	x_train, y_train, x_valid, y_valid = preprocess(path_x, path_y)
	w = np.random.rand(111)
	b = np.random.rand(1)
	train(batch_size, epochs, lr)


