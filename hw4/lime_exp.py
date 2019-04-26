import lime
from lime import lime_image
from keras.models import load_model
import sys
import os
import numpy as np
from skimage.segmentation import mark_boundaries
import skimage
import matplotlib.pyplot as plt

inDrive = False

if inDrive:
	model = load_model(drive_path + 'model_best.h5')
else:
	model = load_model('model_best.h5')

def preprocess(path, start, end):
	x_path = "data/x_train.npy"
	y_path = "data/y_train.npy"
	if (os.path.isfile(x_path) and os.path.isfile(y_path)):
		x_train = np.load(x_path)
		y_train = np.load(y_path)
		x_train = np.reshape(x_train,[-1,48,48])
	else:
		x_train = []
		y_train = []
		with open(path, newline='') as csvfile:
			next(csvfile)
			rows = csv.reader(csvfile)
			for row in rows:
				y_train.append(int(row[0]))
				x_train.append(np.array( row[1].split() ).astype(np.float))

		x_train = np.reshape(np.array(x_train),[-1,48,48])
		y_train = np.array(y_train)

		np.save('data/x_train.npy', x_train)
		np.save('data/y_train.npy', y_train)

	return x_train[start:end+1], y_train[start:end+1]

if __name__ == '__main__':
	path = "data/train.csv"
	if len(sys.argv) > 1:
		start = int(sys.argv[1][1:])
		if len(sys.argv) > 2:
			end = int(sys.argv[2][1:])
		else:
			end = start
	x_train, y_train = preprocess(path, start, end)
	img = skimage.color.gray2rgb(x_train[0])
	# explainer = lime_image.LimeImageExplainer()
	# explanation = explainer.explain_instance(x_train[0], model.predict, top_labels=5, hide_color=0, num_samples=1000)
	# temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=10, hide_rest=False)
	# plt.imshow(mark_boundaries(array_to_img(temp), mask))

# images = [x_train[i] for i in path_list]
# y = [y_train[i] for i in path_list]
# images = np.array(images)
# y = np.array(y)
# explainer = lime_image.LimeImageExplainer()
# predict_ = lambda x : np.squeeze(model.predict(x[:, :, :, 0].reshape(-1, 48, 48, 1)))
# for i in range(7):
#     image = [images[i]] * 3
#     image = np.concatenate(image, axis = 2)
#     np.random.seed(16)
#     explanation = explainer.explain_instance(image, predict_, labels=(i, ), top_labels=None, hide_color=0, num_samples=1000)
#     temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=1000, hide_rest=True)
#     plt.imsave(out_path + 'fig3_' + str(i) + '.jpg', mark_boundaries(temp / 2 + 0.5, mask))