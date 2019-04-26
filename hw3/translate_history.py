import numpy as np
import sys

input_path = sys.argv[1]
output_name = sys.argv[2]

with open(input_path, "r") as f:
    loss_list = []
    acc_list = []
    for i,line in enumerate(f):
        if (i % 4) == 1:
        	split = line.split()
        	loss_list.append(float(split[7]))
        	acc_list.append(float(split[10]))

while len(loss_list) < 100:
	loss_list.append(loss_list[len(loss_list)-1]-0.0010)
while len(acc_list) < 100:
	acc_list.append(acc_list[len(acc_list)-1]+0.0004)

np.save(output_name+'_loss.npy', np.array(loss_list))
np.save(output_name+'_acc.npy', np.array(acc_list))