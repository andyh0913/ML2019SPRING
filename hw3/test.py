import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import csv

lr = 0.0001

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
# dropout1 = tf.nn.dropout(fc1,rate=drop_rate)
fc2 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu)
# dropout2 = tf.nn.dropout(fc2,rate=drop_rate)
dense = tf.layers.dense(fc2, 7)

outputs = tf.nn.softmax(dense)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=dense))
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
		predict = sess.run(outputs, feed_dict={inputs:x_test})
		for i,test_y in enumerate(predict):
			output_list.append([i, np.argmax(test_y)])

	print("Generate ans.csv!")
	output_file = pd.DataFrame(output_list)
	output_file.to_csv(output_path, index=False, header=False)