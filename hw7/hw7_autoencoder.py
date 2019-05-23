import os
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv2D, Conv2DTranspose, Activation, LeakyReLU, Flatten, Reshape, BatchNormalization, Dropout
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

epochs = 100
batch_size = 128
encoded_dim = 192
augmentation = True

def load_data(image_directory='./data/images'):
	if os.path.exists("./train_x.npy"):
		print ("Loading train_x.npy")
		return np.load("./train_x.npy")
	else:
		print ("Loading image data...")
		train_x = []
		for i in range(1,40001):
			# print (i)
			file_path = os.path.join(image_directory, ("%06d" % i) +'.jpg')
			img = Image.open(file_path)
			arr = np.array(img)
			train_x.append(arr)
		train_x = (np.array(train_x)-127.5)/127.5
		np.save("train_x.npy", train_x)
		return train_x # shape = (40000,32,32,3)


if __name__ == '__main__':
	image_directory = './data/images'
	save_directory = './models'
	train_x = load_data()

	if augmentation:
		datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=30, zoom_range=0.2,
				shear_range=0.2, fill_mode='nearest', validation_split=0.1)
		datagen.fit(train_x)

	model = Sequential()
	model.add(Conv2D(128, 5, strides=2, padding='same', input_shape=(32,32,3)))
	model.add(LeakyReLU(0.2))
	model.add(Conv2D(256, 5, strides=2, padding='same'))
	model.add(LeakyReLU(0.2))
	# model.add(BatchNormalization())
	model.add(Conv2D(512, 5, strides=2, padding='same'))
	model.add(LeakyReLU(0.2))
	# model.add(BatchNormalization())
	model.add(Flatten())
	model.add(Dense(encoded_dim))
	####################################
	# TODO: Add activation or not?
	model.add(Dense(4*4*512))
	model.add(LeakyReLU(0.2))
	# model.add(BatchNormalization())
	model.add(Reshape((4,4,512)))
	model.add(Conv2DTranspose(256, 5, strides=2, padding='same'))
	model.add(LeakyReLU(0.2))
	# model.add(BatchNormalization())
	model.add(Conv2DTranspose(128, 5, strides=2, padding='same'))
	model.add(LeakyReLU(0.2))
	# model.add(BatchNormalization())
	model.add(Conv2DTranspose(3, 5, strides=2, padding='same'))
	model.add(Activation('tanh'))

	opt = optimizers.Adam(lr=0.001)
	model.compile(loss='mean_squared_error', optimizer=opt)
	model.summary()

	checkpoint = ModelCheckpoint(os.path.join(save_directory,"model_best.h5"), monitor='val_loss', mode='min', save_best_only=True)

	samples_per_epoch = int(len(train_x)*0.9*8)
	validation_steps = int(len(train_x)*0.1/batch_size)

	if augmentation:
		model.fit_generator(datagen.flow(train_x, train_x, batch_size,subset='training'),
				samples_per_epoch=samples_per_epoch, epochs=epochs,
				validation_data=datagen.flow(train_x, train_x, batch_size,subset='validation'), validation_steps=validation_steps,
				callbacks=[checkpoint])
	else:
		model.fit(train_x, train_x, validation_split=0.2, shuffle=True, callbacks=[checkpoint], epochs=epochs, batch_size=batch_size)

