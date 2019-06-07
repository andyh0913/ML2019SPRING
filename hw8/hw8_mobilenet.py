from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, Softmax, DepthwiseConv2D, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import csv
import numpy as np

batch_size = 256
epochs = 200
valid_rate = 0.1
augmentation = True

def load_data(csv_path="./data/train.csv"):
	print ("Loading training data...")
	x_train = []
	y_train = []
	with open(csv_path, newline='') as csvfile:
		next(csvfile)
		rows = csv.reader(csvfile)
		for row in rows:
			one_hot = np.zeros(7)
			one_hot[int(row[0])] = 1
			y_train.append(one_hot)
			x_train.append(np.array( row[1].split() ).astype(np.float))
	x_train = np.reshape(np.array(x_train),[-1,48,48,1])
	y_train = np.array(y_train)
	return x_train / 255. , y_train

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

	model.summary()
	opt = Adam(lr=0.005)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	return model

if __name__ == '__main__':
	train_data_path = sys.argv[1]
	x_train, y_train = load_data(train_data_path)
	print (x_train.shape)
	model_path = './models/model_best.h5'
	model = build_model()
	checkpoint = ModelCheckpoint(model_path, save_weights_only=True, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	if augmentation:
		datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=30, zoom_range=0.2,
				shear_range=0.2, fill_mode='nearest', validation_split=valid_rate)
		datagen.fit(x_train)
		steps_per_epoch = len(x_train)*(1-valid_rate)*8 // batch_size
		validation_steps = len(x_train)*valid_rate // batch_size
		model.fit_generator(datagen.flow(x_train, y_train, batch_size, shuffle=True, subset='training'),
				steps_per_epoch=steps_per_epoch, epochs=epochs,
				validation_data=datagen.flow(x_train, y_train, batch_size,subset='validation'), 
				validation_steps=validation_steps,
				callbacks=[checkpoint])
	else:
		model.fit(x_train, y_train,
				batch_size=batch_size,
				epochs=epochs,
				shuffle=True,
				validation_split=valid_rate, 
				callbacks=[checkpoint])


