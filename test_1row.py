import numpy as np
import pandas as pd
import sys

if __name__ == "__main__":
	w = np.load('w_1row.npy')
	b = np.load('b_1row.npy')

	if len(sys.argv) == 3:
		input_path = sys.argv[1]
		output_path = sys.argv[2]
	elif len(sys.argv) > 3:
		sys.exit("Extra arguments!")
	else:
		sys.exit("Missing arguments!")
	
	data = np.genfromtxt(input_path, delimiter=',')[:,2:]
	# data = np.delete(data, range(10,data.shape[0], 18), axis=0) # delete RAINFALL (NR)
	for i in range(10, data.shape[0], 18):
		for j in range(data[i].shape[0]):
			if np.isnan(data[i][j]):
				data[i][j] = 0

	n_test = data.shape[0]//18
	
	test_x = []
	for i in range(n_test):
		test_x.append(data[i*18+9])
	output_list = [["id", "value"]]
	for i in range(n_test):
		val = (w * test_x[i]).sum() + b
		output_list.append(["id_"+str(i), val[0]])
	output_file = pd.DataFrame(output_list)
	output_file.to_csv(output_path, index=False, header=False)