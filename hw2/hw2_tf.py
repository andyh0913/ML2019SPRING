import numpy as np
import tensorflow as tf
import sys

# hyper parameters
batch_size = 1000
epochs = 10000
lr = 0.0001
valid_rate = 0.2
keep_prob = 0.5

# model
inputs = tf.placeholder(dtype=tf.float32, shape=(None, 106))
labels = tf.placeholder(dtype=tf.float32, shape=(None, 1))
dense = tf.layers.dense(inputs, 200, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
dense = tf.nn.dropout(dense, keep_prob=keep_prob)
dense = tf.sigmoid(dense)
dense = tf.layers.dense(inputs, 100, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
dense = tf.nn.dropout(dense, keep_prob=keep_prob)
dense = tf.sigmoid(dense)
dense = tf.layers.dense(dense, 20, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
dense = tf.nn.dropout(dense, keep_prob=keep_prob)
dense = tf.sigmoid(dense)
dense = tf.layers.dense(dense, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
outputs = tf.sigmoid(dense)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=dense))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)



# saver
saver = tf.train.Saver()

# raw data
def preprocess(path_x, path_y):
	x_train  = np.genfromtxt(path_x, delimiter=',')[1:]
	y_train = np.genfromtxt(path_y, delimiter=',')[1:]
	# x_train.shape = (32561, 106)
	# y_train.shape = (32561, )
	y_train = np.expand_dims(y_train,1)

	# normalization
	x_mean = x_train.sum(axis=0) / x_train.shape[0]
	x_var = ((x_train - x_mean) ** 2).sum(axis=0) / x_train.shape[0]
	x_train = (x_train - x_mean)/(x_var**0.5)

	np.save('x_mean.npy', x_mean)
	np.save('x_var.npy', x_var)

	n_valid = (int)(x_train.shape[0]*valid_rate)

	return x_train[:-n_valid], y_train[:-n_valid], x_train[-n_valid:], y_train[-n_valid:]

def train(batch_size, epochs):
	global x_train
	global y_train
	global x_val
	global y_val
	iterations = x_train.shape[0] // batch_size

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		min_loss = 1000

		for i in range(epochs):
			assert len(x_train) == len(y_train)
			p = np.random.permutation(len(x_train))
			x_train = x_train[p]
			y_train = y_train[p]

			epoch_loss = 0
			val_loss = 0
			for j in range(iterations):
				x = x_train[j*batch_size:(j+1)*batch_size]
				y = y_train[j*batch_size:(j+1)*batch_size]
				
				_, iter_loss = sess.run([optimizer, loss],feed_dict={inputs:x,labels:y})
				val_loss += sess.run(loss,feed_dict={inputs:x_val,labels:y_val})
				epoch_loss += iter_loss

			epoch_loss /= iterations
			val_loss /= iterations
			print ("{} epoch, loss = {}, val_loss = {}".format(i+1,epoch_loss, val_loss))
			if (val_loss < min_loss):
				saver.save(sess, './model/batch_100'.format(i))
				min_loss = val_loss
		


if __name__ == "__main__":
	path_x = 'data/X_train'
	path_y = 'data/Y_train'
	# data = preprocess(path)
	# print (data.shape)
	x_train, y_train, x_val, y_val = preprocess(path_x, path_y)
	train(batch_size,epochs)



