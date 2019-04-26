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
epsilon = 10

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

def fgsm(x, label):
	sess = K.get_session()
	x = np.expand_dims(x, axis=0)
	y = K.one_hot(np.expand_dims(label,axis=0), 1000)
	loss = K.categorical_crossentropy(y, model.output)
	grads = K.gradients(loss, model.input)
	delta = K.sign(grads[0])
	x_adv = model.input + epsilon*delta

	x_fgsm = sess.run(x_adv, feed_dict={model.input: x})

	preds = model.predict(x_fgsm)
	label = np.expand_dims(label, axis=1)
	correct_pred = K.equal(K.argmax(preds,1), label)
	accuracy = K.mean(K.cast_to_floatx(correct_pred))
	print ("accuracy: ", accuracy)

	return x_fgsm[0]


if __name__ == '__main__':
	x,labels = load_data()
	output_path = "output"
	print (labels.shape)

	for i in range(labels.shape[0]):
		print ("Processing {}-th picture...".format(i))
		img_path = os.path.join(output_path, str(i).zfill(3) + ".png")
		img = fgsm(x[i],labels[i])
		# imsave(img_path, img)
