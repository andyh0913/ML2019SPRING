import numpy as np
import pandas as pd
import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
	file_path = sys.argv[1]
	h = np.load(file_path)[1000:]
	x = np.arange(h.shape[0])
	
	plt.xlabel("epochs")
	plt.ylabel("loss")
	plt.plot(x,h)
	plt.savefig("h_18f.png")