import numpy as np
import matplotlib.pyplot as plt




dnn_train_loss = np.load('rnn_loss.npy')
dnn_train_acc = np.load('rnn_acc.npy')

cnn_train_loss = np.load('rnn_v_loss.npy')
cnn_train_acc = np.load('rnn_v_acc.npy')

it = range(0, dnn_train_loss.shape[0])

plt.plot(it, dnn_train_loss, label='rnn_train_loss')
plt.plot(it, dnn_train_acc, label='rnn_train_acc')
plt.plot(it, cnn_train_loss, label='rnn_valid_loss')
plt.plot(it, cnn_train_acc, label='rnn_valid_acc')

plt.legend(loc='upper right')

# plt.yscale('iteration')
# plt.title('Loss to Iteration')
# plt.yscale('logit')
plt.grid(True)

plt.savefig('rnn_history.png')
plt.show()