import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, Softmax, DepthwiseConv2D, BatchNormalization, GlobalAveragePooling2D

def build_model():
	model = Sequential()
	# (48,48,32)
	model.add(Conv2D(16, 3, strides=2, padding='same', input_shape=(48,48,1)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	# (24,24,32)
	model.add(DepthwiseConv2D(3, strides=1, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(32, 1, strides=1, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	# (24,24,64)
	model.add(DepthwiseConv2D(3, strides=2, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(32, 1, strides=1, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	# (12,12,128)
	model.add(DepthwiseConv2D(3, strides=1, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, 1, strides=1, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	# (12,12,256)
	model.add(DepthwiseConv2D(3, strides=2, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, 1, strides=1, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	
	# (6,6,256)
	model.add(DepthwiseConv2D(3, strides=1, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, 1, strides=1, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	
# 	# (6,6,256)
# 	model.add(DepthwiseConv2D(3, strides=2, padding='same'))
# 	model.add(Activation('relu'))
# 	model.add(BatchNormalization())
# 	model.add(Conv2D(512, 1, strides=1, padding='same'))
# 	model.add(Activation('relu'))
# 	model.add(BatchNormalization())
	# (3,3,512)
	# model.add(DepthwiseConv2D(3, strides=1, padding='same'))
	# model.add(Activation('relu'))
	# model.add(BatchNormalization())
	# model.add(Conv2D(512, 1, strides=1, padding='same'))
	# model.add(Activation('relu'))
	# model.add(BatchNormalization())
	# (3,3,512)
	model.add(GlobalAveragePooling2D())
	# 
	# model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.3))
	model.add(Dense(7))
	model.add(Softmax())
	
	return model

if __name__ == "__main__":
	# compress
	model = build_model()
	model.load_weights('./models/model_best.h5')
	arr = model.get_weights()
	arr_16 = []
	for i in range(len(arr)):
		arr_16.append(arr[i].astype(np.float16))
	print (arr_16[0])
	np.savez_compressed('arr_16.npz', f16=arr_16)

	# load weights
	# arr_16_l = np.load('arr_16.npz')['f16']
	# arr_32 = []
	# for i in range(arr_16_l.shape[0]):
	# 	arr_32.append(arr_16_l[i].astype(np.float32))

	# model = build_model()
	# model.set_weights(arr_32)
	# model.summary()