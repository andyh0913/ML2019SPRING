import os
import sys
import numpy as np 
from skimage.io import imread, imsave

# test_images = ['1.jpg','10.jpg','22.jpg','37.jpg','72.jpg']
k = 5



def load_data_and_normalize(image_dir = './Aberdeen'):
	filelist = [str(i)+'.jpg' for i in range(415)]
	img_shape = imread(os.path.join(image_dir, filelist[0])).shape
	img_data = []
	for filename in filelist:
		tmp = imread(os.path.join(image_dir, filename))
		img_data.append(tmp.flatten())
	img_data = np.array(img_data).astype('float32')
	mean = np.mean(img_data, axis = 0)
	img_data -= mean
	return img_shape, mean, img_data

def process(M): 
	M -= np.min(M)
	M /= np.max(M)
	M = (M * 255).astype(np.uint8)
	return M

if __name__ == '__main__':
	# image_dir = './Aberdeen'
	image_dir = sys.argv[1]
	input_image_name = sys.argv[2]
	output_image_path = sys.argv[3]

	img_shape, mean, training_data = load_data_and_normalize(image_dir)
	# print (training_data)
	# average_img = process(mean)
	
	# problem 1.a
	# imsave('average.jpg', average_img.reshape(img_shape))

	# problem 1.b
	u, s, v = np.linalg.svd(training_data.transpose(), full_matrices = False)
	print (u.shape)
	print (s.shape)
	print (v.shape)

	# for i in range(5):
	# 	eigenface = process(u.transpose()[i])
	# 	imsave(str(i) + '_eigenface.jpg', eigenface.reshape(img_shape))

	# problem 1.c
	test_images = [input_image_name]
	for x in test_images:
		picked_img = imread(os.path.join(image_dir, x))
		X = picked_img.flatten().astype('float32')
		X -= mean

		weight = np.array([X.dot(u.transpose()[i])] for i in range(k))
		reconstruct = process(weight.dot(u.transpose()[:k]) + mean)
		imsave(output_image_path, reconstruct.reshape(img_shape))

	# problem 1.d
	# for i in range(5):
	# 	number = s[i] * 100 / sum(s)
	# 	print(number)