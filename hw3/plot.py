import numpy as np
import matplotlib.pyplot as plt




dnn_train_loss = np.load('dnn_loss.npy')
dnn_train_acc = np.load('dnn_acc.npy')

cnn_train_loss = np.load('cnn_loss.npy')
cnn_train_acc = np.load('cnn_acc.npy')

it = range(0, 100)

plt.plot(it, dnn_train_loss, label='dnn_train_loss')
plt.plot(it, dnn_train_acc, label='dnn_train_acc')
plt.plot(it, cnn_train_loss, label='cnn_train_loss')
plt.plot(it, cnn_train_acc, label='cnn_train_acc')

plt.legend(loc='upper right')

# plt.yscale('iteration')
# plt.title('Loss to Iteration')
# plt.yscale('logit')
plt.grid(True)

plt.savefig('history.png')
plt.show()