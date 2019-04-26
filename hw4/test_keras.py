import numpy as np
import sys
import os
import csv
import keras
import pandas as pd
from keras.models import load_model

drive_path = "/content/drive/My Drive/ML2019Spring/hw3/"
batch_size = 256

inDrive = False

if __name__ == "__main__":

	input_path = sys.argv[1]
	output_path = sys.argv[2]

	if len(sys.argv) > 3:
		if sys.argv[2].strip('-') == 'G' or sys.argv[2].strip('-') == 'g':
			inDrive = True

	x_test = []
	with open(input_path, newline='') as csvfile:
		next(csvfile)
		rows = csv.reader(csvfile)
		for row in rows:
			x_test.append(np.array( row[1].split() ).astype(np.float))

	x_test = np.reshape(np.array(x_test),[-1,48,48,1])
	
	# 0-255 subject to 0-1
	x_test = x_test / 255.

	predict = []
	if iterations*batch_size < x_test.shape[0]:
		x = x_test[iterations*batch_size:]
		predict += np.argmax(model.predict(x), axis=1).tolist()
	for i,test_y in enumerate(predict):
		output_list.append([i, test_y])

	print("Generate ans.csv!")
	output_file = pd.DataFrame(output_list)
	output_file.to_csv(output_path, index=False, header=False)