import numpy as np
import sys
import os
import csv
import keras
import pandas as pd
from keras.models import load_model

batch_size = 256

if __name__ == "__main__":

	# input_path = sys.argv[1]
	# output_path = sys.argv[2]
	input_path = "./data/test.csv"
	output_path = "./result/ans.csv"

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
	model = load_model('./models/model_best.h5')
	model.save_weights('./models/model_best_weights.h5')
	
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