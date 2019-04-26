import numpy as np
from scipy.misc import imsave
from keras.applications import vgg16, resnet50
from keras.preprocessing import image
# tensorflow backend
import keras.backend as K
import sys
import os

# model = vgg16.VGG16(weights='imagenet')
model = resnet50.ResNet50(weights='imagenet')
epsilon = 0.1
epochs = 100
batch_size = 1

def load_data(folder_path="data/images", labels_path="data/labels.csv"):
	if os.path.isfile("data/img_list.npy"):
		img_list = np.load("data/img_list.npy")
	else:
		img_list = []
		for i in range(200):
			img_path = os.path.join(folder_path, str(i).zfill(3) + ".png")
			img = image.load_img(img_path, target_size=(224,224))
			img_list.append(image.img_to_array(img))
		img_list = np.array(img_list)
		np.save("data/img_list.npy", img_list)
	labels = np.genfromtxt(labels_path, delimiter=',')[1:,3:4]
	return img_list, labels

def fgsm(x, labels):
	sess = K.get_session()
	y = K.one_hot(labels, 1000)
	loss = K.categorical_crossentropy(y, model.output)
	grads = K.gradients(loss, model.input)
	delta = K.sign(grads)
	x_adv = model.input + epsilon*delta[0]
	
	x_iter = x
	for i in range(epochs):
		x_iter = sess.run(x_adv, feed_dict={model.input: x_iter})
		print ("Epoch",i)

	# correct_pred = K.equal(K.argmax(preds,1), labels)
	# accuracy = K.mean(K.cast_to_floatx(correct_pred))

	return x_iter[0]


if __name__ == '__main__':
	x,labels = load_data()
	output_path = "output"
	print (labels.shape)
	img = fgsm(x,labels)

	iterations = x.shape[0] // batch_size
	for i in range(iterations):
		imgs = fgsm(x[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])
		for j in range(batch_size):
			img_path = os.path.join(output_path, str(i*iterations+j).zfill(3) + ".png")
			imsave(img_path, imgs[j])
