import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import csv

drive_path = "/content/drive/My Drive/ML2019Spring/"

lr = 0.001
batch_size = 256

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
# model = drop(model)
model = fc(model, 512)
# model = drop(model)
model = fc(model, 7)

outputs = tf.nn.softmax(model)

correct_pred = tf.equal(tf.argmax(outputs, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=model))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
if __name__ == "__main__":

	input_path = sys.argv[1]
	output_path = sys.argv[2]

	x_test = []
	with open(input_path, newline='') as csvfile:
		next(csvfile)
		rows = csv.reader(csvfile)
		for row in rows:
			x_test.append(np.array( row[1].split() ).astype(np.float))

	x_test = np.reshape(np.array(x_test),[-1,48,48,1])
	
	# 0-255 subject to 0-1
	x_test = x_test / 255.

	output_list = [["id", "label"]]
	saver = tf.train.Saver()
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state('model')
		if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
			saver.restore(sess, ckpt.model_checkpoint_path)
		
		iterations =  x_test.shape[0]//batch_size
		predict = []
		for j in range(iterations):
			x = x_test[j*batch_size:(j+1)*batch_size]
			predict += list( sess.run(outputs, feed_dict={inputs:x}) )
		if iterations*batch_size != x_test.shape[0]:
			predict += list( sess.run(outputs, feed_dict={inputs:x_test[iterations*batch_size:]}) )
		for i,test_y in enumerate(predict):
			output_list.append([i, np.argmax(test_y)])

	print("Generate ans.csv!")
	output_file = pd.DataFrame(output_list)
	output_file.to_csv(output_path, index=False, header=False)