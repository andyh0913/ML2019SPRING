import numpy as np
import tensorflow as tf
import sys
import csv

# hyper parameters
batch_size = 256
epochs = 100
lr = 0.0001
valid_rate = 0.2
drop_rate = 0.4

# model
inputs = tf.placeholder(dtype=tf.float32, shape=(None, 48, 48, 1))
labels = tf.placeholder(dtype=tf.float32, shape=(None, 7))

conv1 = tf.layers.conv2d(inputs, 32, [5,5],padding='same',activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, [3,3], [2,2])

conv2 = tf.layers.conv2d(pool1, 32, [4,4],padding='same',activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, [3,3], [2,2])

conv3 = tf.layers.conv2d(pool2, 64, [5,5],padding='same',activation=tf.nn.relu)
pool3 = tf.layers.max_pooling2d(conv3, [3,3], [2,2])

flat = tf.layers.flatten(pool3)
fc1 = tf.layers.dense(flat, 2048, activation=tf.nn.relu)
dropout1 = tf.nn.dropout(fc1,rate=drop_rate)
fc2 = tf.layers.dense(dropout1, 1024, activation=tf.nn.relu)
dropout2 = tf.nn.dropout(fc2,rate=drop_rate)
dense = tf.layers.dense(dropout2, 7)

outputs = tf.nn.softmax(dense)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=dense))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)



# saver
saver = tf.train.Saver()

# raw data
def preprocess(path):
	x_train = []
	y_train = []
	with open(path, newline='') as csvfile:
		next(csvfile)
		rows = csv.reader(csvfile)
		for row in rows:
			one_hot = np.zeros(7)
			one_hot[int(row[0])] = 1
			y_train.append(one_hot)
			x_train.append(np.array( row[1].split() ).astype(np.float))

	# x_train.shape = (28709, 48*48)
	# y_train.shape = (28709, 7)
	x_train = np.reshape(np.array(x_train),[-1,48,48,1])
	y_train = np.array(y_train)

	# 0-255 subject to 0-1
	x_train = x_train / 255.

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
				print ("Save model!")
				saver.save(sess, '/content/drive/My Drive/ML2019Spring/model/batch_256')
				min_loss = val_loss
		


if __name__ == "__main__":
	data_path = '/content/drive/My Drive/ML2019Spring/data/train.csv'
	x_train, y_train, x_val, y_val = preprocess(data_path)
	train(batch_size,epochs)



