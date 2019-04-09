import numpy as np
import sys
import os
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

drive_path = "/content/drive/My Drive/ML2019Spring/hw3/"
# drive_path + 

inDrive = False
model_num = -1

valid_rate = 0.2
epochs = 100
batch_size = 256 

def preprocess(path):
	if inDrive:
		x_path = drive_path + "data/x_train.npy"
		y_path = drive_path + "data/y_train.npy"
	else:
		x_path = "data/x_train.npy"
		y_path = "data/y_train.npy"
	
	if (os.path.isfile(x_path) and os.path.isfile(y_path)):
		x_train = np.load(x_path)
		y_train = np.load(y_path)
		n_valid = (int)(x_train.shape[0]*valid_rate)
	else:
		x_train = []
		y_train = []
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
		x_train = np.reshape(np.array(x_train),[-1,48,48,1])
		y_train = np.array(y_train)

		# 0-255 subject to 0-1
		x_train = x_train / 255.
		n_valid = (int)(x_train.shape[0]*valid_rate)
		if inDrive:
			np.save(drive_path + 'data/x_train.npy', x_train)
			np.save(drive_path + 'data/y_train.npy', y_train)
		else:
			np.save('data/x_train.npy', x_train)
			np.save('data/y_train.npy', y_train)

	return x_train[:-n_valid], y_train[:-n_valid], x_train[-n_valid:], y_train[-n_valid:]

def train(epochs, batch_size, argumentation=False):
	global x_train
	global y_train
	global x_valid
	global y_valid

	#data argumentation
	if argumentation:
		datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=30, zoom_range=0.2,
				shear_range=0.2, fill_mode='nearest')
		datagen.fit(x_train)
	#model
	models = [Sequential(),Sequential(),Sequential(),Sequential(),Sequential()]

	model_struct = [
		[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
		[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
		[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
		[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
		[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
	]

	i = model_num
	for j, c in enumerate(model_struct[i]):
		if c == 'M':
			models[i].add(MaxPooling2D(pool_size=2))
		elif j == 0:
			models[i].add(Conv2D(filters=c, kernel_size=3, padding='same', activation='relu', input_shape=(48, 48, 1)))
			models[i].add(BatchNormalization())
		else:
			models[i].add(Conv2D(filters=c, kernel_size=3, padding='same', activation='relu'))
			models[i].add(BatchNormalization())
	models[i].add(Flatten())
	models[i].add(Dense(256, activation='relu'))
	models[i].add(Dropout(0.5))
	models[i].add(Dense(256, activation='relu'))
	models[i].add(Dropout(0.5))
	models[i].add(Dense(7, activation='softmax'))
	print (models[i])
	models[i].summary()
	#compiling
	adam = Adam(lr=0.0001)
	models[i].compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	
	#check point
	if inDrive:
		filepath=drive_path + 'model_'+str(i)+'.h5'
	else:
		filepath='model_'+str(i)+'.h5'

	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	
	#fitting params
	samples_per_epoch = len(y_train)*4
	#fitting
	if argumentation:
		models[i].fit_generator(datagen.flow(x_train, y_train, batch_size),
				samples_per_epoch=samples_per_epoch, epochs=epochs,
				validation_data=(x_valid, y_valid),
				callbacks=callbacks_list)
	else:
		models[i].fit(x_train, y_train,
				batch_size=batch_size,
				epochs=epochs,
				validation_data=(x_valid, y_valid), 
				callbacks=callbacks_list)

if __name__=="__main__":

	if len(sys.argv) > 2:
		if sys.argv[2].strip('-') == 'G' or sys.argv[2].strip('-') == 'g':
			inDrive = True
	if len(sys.argv) > 1:
		model_num = int(sys.argv[1].strip('-'))

	if inDrive:
		path = drive_path + "data/train.csv"
		print ("Train in google drive...")
	else:
		print ("Train in local...")
		path = "data/train.csv"
	x_train, y_train, x_valid, y_valid = preprocess(path)
	train(epochs, batch_size, True)