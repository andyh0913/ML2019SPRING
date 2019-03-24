import numpy as np
import pandas as pd
import sys

def sigmoid(x):
	return 1/(1+np.exp(-x))

if __name__ == "__main__":
	input_path = sys.argv[2]
	output_path = sys.argv[6]
	
	test_x = np.genfromtxt(input_path, delimiter=',')[1:]

	sigma = np.load('sigma.npy')
	mu = np.load('mu.npy')
	num = np.load('num.npy')

	sigma_inv = np.linalg.pinv(sigma)
	w = np.dot((mu[1]-mu[0]), sigma_inv)
	x = test_x.T
	b = (-0.5)*np.dot(np.dot([mu[1]], sigma_inv), mu[1]) \
		+(0.5)*np.dot(np.dot([mu[0]], sigma_inv), mu[0]) \
		+np.log(float(num[1])/num[0])
	z = np.dot(w, x) + b
	output = sigmoid(z)

	output_list = [["id", "label"]]
	for i in range(output.shape[0]):
		if (output[i]>0.5):
			output_list.append([i+1, 1])
		else:
			output_list.append([i+1,0])

		
	output_file = pd.DataFrame(output_list)
	output_file.to_csv(output_path, index=False, header=False)
	print ("Generate ans.csv!")