import numpy as np
from scipy.misc import imsave
from keras.applications import vgg16, resnet50, vgg19
from keras.preprocessing import image
# tensorflow backend
import keras.backend as K
import sys
import os

# model = vgg16.VGG16(weights='imagenet')
# model = vgg19.VGG19(weights='imagenet')
model = resnet50.ResNet50(weights='imagenet')
epsilon = 30

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
    output_path = "output"
    sess = K.get_session()
    y = K.one_hot(np.expand_dims(labels,axis=0), 1000)
    loss = K.categorical_crossentropy(y, model.output)
    grads = K.gradients(loss, model.input)
    delta = K.sign(grads[0])
    x_adv = model.input + epsilon*delta
    correct_preds = 0
    
    for i in range(labels.shape[0]):
        print ("processing "+str(i)+"th image")
        x_fgsm = sess.run(x_adv, feed_dict={model.input: np.expand_dims(x[i], axis=0)})
        img_path = os.path.join(output_path, str(i).zfill(3) + ".png")
        bgr_avr = np.array([103.939, 116.779, 123.68])
        bgr_avr = bgr_avr[np.newaxis,np.newaxis,:]
        img = x_fgsm[0] + bgr_avr
        img[:,:,(0,1,2)] = img[:,:,(2,1,0)]
        img = np.clip(img,0,255)
        imsave(img_path, img)

        preds = model.predict(x_fgsm)
        preds = np.argmax(preds, axis=1)
        if(preds[0]==int(labels[i])):
            correct_preds += 1

    return 1 - correct_preds / labels.shape[0]


if __name__ == '__main__':
    x,labels = load_data()
    x = resnet50.preprocess_input(x)
    success_rate = fgsm(x, labels)
    print ("success rate:", success_rate)	

    
