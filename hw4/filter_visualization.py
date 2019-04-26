import numpy as np
import sys
import os
import csv
import keras
import pandas as pd
from keras.models import load_model
from keras import backend as K
from PIL import Image
import tensorflow as tf


drive_path = "/content/drive/My Drive/ML2019Spring/hw4/"
batch_size = 256
epochs = 100000
lr = 1

inDrive = False

if inDrive:
	model = load_model(drive_path + 'model_best.h5')
else:
	model = load_model('model_best.h5')

def layer_output(x, layer_n, filter_n):
	sess = K.get_session()
	output = model.layers[layer_n].output[:,:,:,filter_n]
	grad_sum = 0.000001
	last_loss = -100
	out = sess.run(output, feed_dict={model.input: x[np.newaxis,:]})
	out = (out * 255 / np.max(out)).astype(np.uint8)
	img = Image.fromarray(out[0])
	img.save(str(filter_n).zfill(3)+".png")

def gradient_ascent(layer_n, filter_n):
	sess = K.get_session()
	output = model.layers[layer_n].output[:,:,:,filter_n]
	loss = K.mean(output)
	grads = K.gradients(loss, model.input)
	print (grads)
	x = np.random.random_sample(model.input.shape[1:])
	print (x.shape)
	grad_sum = 0.000001
	last_loss = -100
	for i in range(epochs):
		e_loss, grad = sess.run([loss, grads], feed_dict={model.input: x[np.newaxis,:]})
		if i % 100 == 0:
			print ("epoch ", i, end=' ')
			print ("loss = ", e_loss)
		if (e_loss - last_loss) < 0.0001:
			break
		last_loss = e_loss
		grad_sum += grad[0][0] ** 2
		x = x + lr * grad[0][0] / (grad_sum) ** 0.5

	positive_x = x - np.amin(x, axis=(1,2), keepdims=True)
	x = (positive_x * 255 / np.amax(positive_x, axis=(1,2), keepdims=True)).astype(np.uint8)

	img = Image.fromarray(np.squeeze(x, axis=-1))
	img.save(str(filter_n).zfill(3)+".png")




if __name__ == "__main__":
	for i,layer in enumerate(model.layers):
		print (i, layer)
	layer_n = input("Please enter the index of the layer: ")
	filter_n = input("please enter the index of the filters (e.g. 0:32): ")
	layer_n = int(layer_n)
	filter_n = [int(f) for f in filter_n.split(':')]
	x = np.load('5286.npy')

	print (layer_n)
	print (filter_n)
	# for i in range(filter_n[0],filter_n[1]):
	# 	gradient_ascent(layer_n,i)
	for i in range(filter_n[0],filter_n[1]):
		layer_output(x,layer_n,i)