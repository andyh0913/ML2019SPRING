import numpy as np
import pandas as pd
import tensorflow as tf
import sys

lr = 0.0001

# model
inputs = tf.placeholder(dtype=tf.float32, shape=(None, 106))
labels = tf.placeholder(dtype=tf.float32, shape=(None, 1))
dense = tf.layers.dense(inputs, 200, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
dense = tf.sigmoid(dense)
dense = tf.layers.dense(dense, 100, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
dense = tf.sigmoid(dense)
dense = tf.layers.dense(dense, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
outputs = tf.sigmoid(dense)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=dense))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

if __name__ == "__main__":
	
	if len(sys.argv) == 3:
		input_path = sys.argv[1]
		output_path = sys.argv[2]
	elif len(sys.argv) > 3:
		sys.exit("Extra arguments!")
	else:
		sys.exit("Missing arguments!")

	data = np.genfromtxt(input_path, delimiter=',')[1:]

	# normalization
	x_mean = np.load('x_mean.npy')
	x_var = np.load('x_var.npy')
	data = (data - x_mean)/(x_var**0.5)

	output_list = [["id", "label"]]
	saver = tf.train.Saver()
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state('model')
		if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
			saver.restore(sess, ckpt.model_checkpoint_path)
		predict = sess.run(outputs, feed_dict={inputs:data})
		for i,test_y in enumerate(predict):
			if (test_y>0.5):
				output_list.append([i+1, 1])
			else:
				output_list.append([i+1,0])

	print("Generate ans.csv!")
	output_file = pd.DataFrame(output_list)
	output_file.to_csv(output_path, index=False, header=False)