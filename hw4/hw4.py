import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import load_model
import sys
import os
import csv
from skimage.segmentation import mark_boundaries
import skimage
from lime import lime_image
from skimage.segmentation import slic

model = load_model('model.h5')

indexes = [5286, 27925, 5173, 24476, 10383, 25580, 28364]
lr = 1

def preprocess(path):
	x_train = []
	y_train = []
	print ("Reading csv file...")
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
	x_train = np.reshape(np.array(x_train),[-1,48,48,1]) / 255.
	y_train = np.array(y_train)

	print ("File read finish!")
	return x_train, y_train


def saliency_map(x):
	global output_path
	sess = K.get_session()
	output = K.max(model.layers[22].output, axis = 1)
	# print (output)
	grads = K.gradients(output, model.input)
	# print (grads)
	w = K.abs(grads)[0]
	# print (w)
	out, grad, w = sess.run([output, grads, w], feed_dict={model.input: x})
	w = w / np.amax(w,axis=(1,2,3),keepdims=True)
	for i in range(w.shape[0]):
		z = np.squeeze(w[i], axis=-1)
		fig, ax = plt.subplots()
		cax = ax.imshow(z, cmap = 'jet')
		fig.colorbar(cax, ax=ax)
		plt.savefig(output_path + 'fig1_'+str(i)+'.jpg')
	return

def layer_output(x, layer_n):
	global output_path
	for filter_n in range(64):
		sess = K.get_session()
		output = model.layers[layer_n].output[:,:,:,filter_n]
		out = sess.run(output, feed_dict={model.input: x[np.newaxis,:]})
		# out = (out * 255 / np.max(out)).astype(np.uint8)

		plt.figure(num = 'filters', figsize = (8, 8))
		plt.subplot(8, 8, filter_n+1)
		plt.axis('off')
		plt.imshow(out[0].reshape(48, 48), cmap = 'gray')
	plt.savefig(output_path + 'fig2_2.jpg')

def gradient_ascent(layer_n):
	global output_path
	for filter_n in range(64):
		sess = K.get_session()
		output = model.layers[layer_n].output[:,:,:,filter_n]
		loss = K.mean(output)
		grads = K.gradients(loss, model.input)
		x = np.random.random_sample(model.input.shape[1:])
		grad_sum = 0.000001
		last_loss = -100
		for i in range(100):
			e_loss, grad = sess.run([loss, grads], feed_dict={model.input: x[np.newaxis,:]})
			last_loss = e_loss
			grad_sum += grad[0][0] ** 2
			x = x + lr * grad[0][0] / (grad_sum) ** 0.5

		x = x - np.amin(x, axis=(1,2), keepdims=True)
		# x = (positive_x * 255 / np.amax(positive_x, axis=(1,2), keepdims=True)).astype(np.uint8)

		plt.figure(num = 'filters', figsize = (8, 8))
		plt.subplot(8, 8, filter_n+1)
		plt.axis('off')
		plt.imshow(x.reshape(48, 48), cmap = 'gray')
	plt.savefig(output_path + 'fig2_1.jpg')

def LIME(model, x, i):
	global output_path
	explainer = lime_image.LimeImageExplainer()
	# print('x_size = ', x.shape)
	predict = lambda input: (model.predict(input[:,:,:,0].reshape(-1, 48, 48, 1)))
	# predict = lambda input: print(input)
	img = skimage.color.gray2rgb(x.reshape(48, 48))
	# print('img_size = ', img.shape)
	explanation = explainer.explain_instance(img, predict, labels = (i, ), top_labels=None, hide_color=0, num_samples=1000, segmentation_fn=segmentation)
	temp, mask = explanation.get_image_and_mask(i,positive_only=False,
                                hide_rest=False,
                                num_features=5,
                                min_weight=0.0)
	plt.imsave(output_path + 'fig3_' + str(i) + '.jpg', mark_boundaries(temp, mask).reshape((48, 48, 3)), cmap = 'jet')

def segmentation(img):
	segments = slic(img, n_segments=100, compactness=10)
	return segments

if __name__ == '__main__':
	input_path = sys.argv[1]
	output_path = sys.argv[2]

	x_train, y_train = preprocess(input_path)

	images = [x_train[i] for i in indexes]
	images = np.array(images)
	saliency_map(images)
	gradient_ascent(2)
	layer_output(x_train[5286],2)
	for i in range(7):
		LIME(model, x_train[indexes[i]], i)



