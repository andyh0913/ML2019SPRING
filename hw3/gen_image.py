from PIL import Image
import numpy as np
import sys
import csv

data = np.zeros((48, 48), dtype=np.uint8)
input_path = sys.argv[1]
data = []
with open(input_path, newline='') as csvfile:
	next(csvfile)
	rows = csv.reader(csvfile)
	for row in rows:
		data.append(np.array( row[1].split() ).astype(np.uint8))
	data = np.reshape(np.array(data),[-1,48,48])
for i,img_arr in enumerate(data):	
	img = Image.fromarray(img_arr)
	img.save('train_image/img{}.png'.format(i))
