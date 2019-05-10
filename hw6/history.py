import numpy as np
import sys

input_path = sys.argv[1]
output_name = sys.argv[2]

with open(input_path, "r") as f:
    loss_list = []
    acc_list = []
    v_loss_list = []
    v_acc_list = []
    for i,line in enumerate(f):
        if (i % 2) == 1:
        	split = line.split()
        	# print (split)
        	loss_list.append(float(split[7]))
        	acc_list.append(float(split[10]))
        	v_loss_list.append(float(split[13]))
        	v_acc_list.append(float(split[16]))

np.save(output_name+'_loss.npy', np.array(loss_list))
np.save(output_name+'_acc.npy', np.array(acc_list))
np.save(output_name+'_v_loss.npy', np.array(v_loss_list))
np.save(output_name+'_v_acc.npy', np.array(v_acc_list))