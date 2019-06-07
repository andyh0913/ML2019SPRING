import numpy as np
import sys
import os
import csv
import keras
import pandas as pd
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, Softmax, DepthwiseConv2D, BatchNormalization, GlobalAveragePooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

seed = 312
np.random.seed(seed)

batch_size = 256

def build_model():
	model = Sequential()

	model.add(Conv2D(16, 3, strides=2, padding='same', input_shape=(48,48,1)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(DepthwiseConv2D(3, strides=1, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(32, 1, strides=1, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(DepthwiseConv2D(3, strides=2, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(32, 1, strides=1, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(DepthwiseConv2D(3, strides=1, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, 1, strides=1, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(DepthwiseConv2D(3, strides=2, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, 1, strides=1, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(DepthwiseConv2D(3, strides=1, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, 1, strides=1, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(GlobalAveragePooling2D())

	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.3))
	model.add(Dense(7))
	model.add(Softmax())

	return model

if __name__ == "__main__":
	augmentation = True
	ensemble_num = 8
	input_path = sys.argv[1]
	output_path = sys.argv[2]
	# input_path = "./data/test.csv"
	# output_path = "./result/ans_new_c.csv"

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
	model = build_model()
	# load weights
	arr_16_l = np.load('arr_16.npz')['f16']
	arr_32 = []
	for i in range(arr_16_l.shape[0]):
		arr_32.append(arr_16_l[i].astype(np.float32))

	model.set_weights(arr_32)
	model.summary()
# 	model.load_weights('./models/model_best.h5')
	
	if augmentation:
		datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=30, zoom_range=0.2,
				shear_range=0.2, fill_mode='nearest')
		steps = len(x_test) // batch_size + 1
		model.pop() # take away softmax layer
		predicts = np.zeros((7178,7))
		for i in range(ensemble_num):
			seed += 1
			datagen.fit(x_test, seed=seed)
			preds_list = []
			j = 0
			for x_batch in datagen.flow(x_test, batch_size=batch_size, shuffle=False, seed=seed):
				j += 1
				preds = model.predict(x_batch)
				preds_list.append(preds)
				if j == steps: break
			predicts += np.concatenate(preds_list, axis=0)
		predicts = np.argmax(predicts, axis = -1)
		output_file = pd.DataFrame(predicts, columns=['label'])
		new_ids=output_file.index.set_names("id")
		output_file.index=new_ids
		output_file.to_csv(output_path, index=True, header=True)
		print ("Generate ans.csv!")
	else:
		iterations =  x_test.shape[0]//batch_size
		predict = []
		for j in range(iterations):
			x = x_test[j*batch_size:(j+1)*batch_size]
			predict += np.argmax(model.predict(x), axis=1).tolist()
		if iterations*batch_size < x_test.shape[0]:
			x = x_test[iterations*batch_size:]
			predict += np.argmax(model.predict(x), axis=1).tolist()
		for i,test_y in enumerate(predict):
			output_list.append([i, test_y])

		print("Generate ans.csv!")
		output_file = pd.DataFrame(output_list)
		output_file.to_csv(output_path, index=False, header=False)