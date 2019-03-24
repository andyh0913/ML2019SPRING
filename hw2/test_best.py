import numpy as np
import pandas as pd
import tensorflow as tf
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation, LeakyReLU, BatchNormalization
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Dense(input_dim=106, units=1000))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
model.add(Dense(units=500))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
model.add(Dense(units=1))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
model.load_weights("weights.best.hdf5")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

if __name__ == "__main__":
	
	if len(sys.argv) == 3:
		input_path = sys.argv[1]
		output_path = sys.argv[2]
	elif len(sys.argv) > 3:
		sys.exit("Extra arguments!")
	else:
		sys.exit("Missing arguments!")

	data = np.genfromtxt(input_path, delimiter=',')[1:]

	test_y = model.predict(data, batch_size=data.shape[0],verbose=0)
	test_y = np.squeeze(test_y)

	output_list = [["id", "label"]]
	for i,test_x in enumerate(data):
		if (test_y[i]>0.5):
			output_list.append([i+1, 1])
		else:
			output_list.append([i+1,0])

	print("Generate ans.csv!")
	output_file = pd.DataFrame(output_list)
	output_file.to_csv(output_path, index=False, header=False)