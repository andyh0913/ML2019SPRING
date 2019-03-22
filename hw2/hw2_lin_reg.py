import numpy as np
import sys

batch_size = 100 # 5652 = 471 * 12
epochs = 100000
lr = 0.01
lamb = 0.95

# raw data
def preprocess(path_x, path_y):
	x_train  = np.genfromtxt(path_x, delimiter=',')[1:]
	y_train = np.genfromtxt(path_y, delimiter=',')[1:]
	# x_train.shape = (32561, 106)
	# y_train.shape = (32561, )
	return x_train, y_train

def train(batch_size, epochs, lr):
	global x_train
	global y_train
	global w
	global b

	iterations = x_train.shape[0] // batch_size

	prev_grad_w = 0.000001
	prev_grad_b = 0.000001

	prev_loss = 0

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
		print ("{} epoch, loss = {}".format(i+1,loss))
		# if abs(loss - prev_loss) < 0.00001:
		# 	print ("Training finished")
		# 	return
		prev_loss = loss


if __name__ == "__main__":
	path_x = 'data/X_train'
	path_y = 'data/Y_train'
	# data = preprocess(path)
	# print (data.shape)
	x_train, y_train = preprocess(path_x, path_y)
	w = np.random.rand(106)
	b = np.random.rand(1)
	try:
		train(batch_size, epochs, lr)
	except (KeyboardInterrupt, SystemExit):
		print ("save model!")
		np.save('w.npy',w)
		np.save('b.npy',b)
		raise
	
	print ("save model!")
	np.save('w.npy',w)
	np.save('b.npy',b)

