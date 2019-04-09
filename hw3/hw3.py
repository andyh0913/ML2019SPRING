import numpy as np
import tensorflow as tf
import sys
import csv

drive_path = "/content/drive/My Drive/ML2019Spring/"

# hyper parameters
batch_size = 256
epochs = 100
lr = 0.001
valid_rate = 0.2
drop_rate = 0.5

def conv(x, filters):
	return tf.keras.layers.Conv2D(filters,3,padding='same',activation='relu')(x)

def maxpool(x):
	return tf.keras.layers.MaxPooling2D(2)(x)

def fc(x, units):
	return tf.keras.layers.Dense(units, activation='relu')(x)

def drop(x):
	return tf.nn.dropout(x,rate=drop_rate)

def flat(x):
	return tf.keras.layers.Flatten()(x)



# model
inputs = tf.placeholder(dtype=tf.float32, shape=(None, 48, 48, 1))
labels = tf.placeholder(dtype=tf.int32, shape=(None, 7))

model = conv(inputs, 64)
model = conv(model, 64)
model = maxpool(model)
model = conv(model, 128)
model = conv(model, 128)
model = maxpool(model)
model = conv(model, 256)
model = conv(model, 256)
model = maxpool(model)
model = conv(model, 512)
model = conv(model, 512)
model = maxpool(model)
model = flat(model)
model = fc(model, 512)
model = drop(model)
model = fc(model, 512)
model = drop(model)
model = fc(model, 7)

outputs = tf.nn.softmax(model)

correct_pred = tf.equal(tf.argmax(outputs, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=model))
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
	val_iterations = x_val.shape[0] // batch_size

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		max_acc = 0

		for i in range(epochs):
			assert len(x_train) == len(y_train)
			p = np.random.permutation(len(x_train))
			x_train = x_train[p]
			y_train = y_train[p]

			epoch_loss = 0
			epoch_acc = 0
			for j in range(iterations):
				x = x_train[j*batch_size:(j+1)*batch_size]
				y = y_train[j*batch_size:(j+1)*batch_size]
				
				_, iter_loss, iter_acc = sess.run([optimizer, loss, accuracy],feed_dict={inputs:x,labels:y})
				
				epoch_loss += iter_loss
				epoch_acc += iter_acc
			epoch_loss /= iterations
			epoch_acc /= iterations

			val_loss = 0
			val_acc = 0
			for j in range(val_iterations):
				x = x_val[j*batch_size:(j+1)*batch_size]
				y = y_val[j*batch_size:(j+1)*batch_size]
				
				iter_val_loss, iter_val_acc = sess.run([loss, accuracy],feed_dict={inputs:x,labels:y})

				val_loss += iter_val_loss
				val_acc += iter_val_acc
			val_loss /= val_iterations
			val_acc /= val_iterations

			print ("{} epoch, train_loss = {}, train_acc = {}, val_loss = {}, val_acc = {}".format(i+1,epoch_loss, epoch_acc, val_loss, val_acc))
			if (val_acc >= max_acc):
				print ("Save model!")
				saver.save(sess, '/content/drive/My Drive/ML2019Spring/model/batch_256')
				max_acc = val_acc
		


if __name__ == "__main__":
	data_path = '/content/drive/My Drive/ML2019Spring/data/train.csv'
	x_train, y_train, x_val, y_val = preprocess(data_path)
	train(batch_size,epochs)



