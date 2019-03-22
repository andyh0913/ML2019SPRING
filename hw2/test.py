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
	
	data = np.genfromtxt(input_path, delimiter=',')[1:]
	# data = np.delete(data, range(10,data.shape[0], 18), axis=0) # delete RAINFALL (NR)

	output_list = [["id", "label"]]
	for i,test_x in enumerate(data):
		z = (test_x * w).sum() + b
		y_h = 1 / ( 1 + np.exp(-z) )
		if (np.asscalar(y_h)>0.5):
			output_list.append([i+1, 1])
		else:
			output_list.append([i+1,0])

		
	output_file = pd.DataFrame(output_list)
	output_file.to_csv(output_path, index=False, header=False)