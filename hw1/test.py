import numpy as np
import pandas as pd
import sys

if __name__ == "__main__":
	w = np.load('w.npy')
	b = np.load('b.npy')

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

	for i in range(data.shape[0]):
		for j in range(9):
			if data[i][j] < 0:
				l, r = 0, 0
				if j == 0 or j == 8:
					data[i][j] = 0
					continue
				else:
					l = j-1
					r = j+1
					while(data[i][l]<0):
						l -= 1
						if l < 0:
							data[i][0] = 0
							l = 1
							break
					while(data[i][r]<0):
						r += 1
						if r > 8:
							data[i][8] = 0
							r = 7
							break
					delta = (data[i][r] - data[i][l])/(r - l)
					data[i][l:r] = [ data[i][l] + (x - l)*delta for x in range(l,r) ]


	n_test = data.shape[0]//18
	
	test_x = []
	for i in range(n_test):
		test_x.append(data[i*18:(i+1)*18])
	output_list = [["id", "value"]]
	for i in range(n_test):
		val = (w * test_x[i]).sum() + b
		output_list.append(["id_"+str(i), val[0]])
	output_file = pd.DataFrame(output_list)
	output_file.to_csv(output_path, index=False, header=False)