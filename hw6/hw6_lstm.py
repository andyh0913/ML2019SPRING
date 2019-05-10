import pandas as pd
import numpy as np
import os
import jieba
from gensim.models import Word2Vec
from gensim import models
from gensim.models import KeyedVectors
import json
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Bidirectional, LeakyReLU, multiply, Layer
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import optimizers
from attention import Attention

# import heapq

max_time_steps = 100
embedding_dim = 100
epochs = 100
batch_size = 1000

rate = 0.5

# Constants
PAD = 0
UNK = 1

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

class LSTMPeepholeCell(LSTM):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(LSTMPeepholeCell, self).__init__(
            units=units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            **kwargs
        )

    def build(self, input_shape):
        super(LSTMPeepholeCell, self).build(input_shape)
        self.recurrent_kernel_c = K.zeros_like(self.recurrent_kernel_c)

    def step(self, inputs, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        dp_mask = states[2]
        rec_dp_mask = states[3]

        if self.implementation == 2:
            z = K.dot(inputs * dp_mask[0], self.kernel)
            z += K.dot(c_tm1 * rec_dp_mask[0], self.recurrent_kernel)
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]
            z2 = z[:, 2 * self.units: 3 * self.units]
            z3 = z[:, 3 * self.units:]

            i = self.recurrent_activation(z0)
            f = self.recurrent_activation(z1)
            c = f * c_tm1 + i * self.activation(z2)
            o = self.recurrent_activation(z3)
        else:
            if self.implementation == 0:
                x_i = inputs[:, :self.units]
                x_f = inputs[:, self.units: 2 * self.units]
                x_c = inputs[:, 2 * self.units: 3 * self.units]
                x_o = inputs[:, 3 * self.units:]
            elif self.implementation == 1:
                x_i = K.dot(inputs * dp_mask[0], self.kernel_i) + self.bias_i
                x_f = K.dot(inputs * dp_mask[1], self.kernel_f) + self.bias_f
                x_c = K.dot(inputs * dp_mask[2], self.kernel_c) + self.bias_c
                x_o = K.dot(inputs * dp_mask[3], self.kernel_o) + self.bias_o
            else:
                raise ValueError('Unknown `implementation` mode.')

            i = self.recurrent_activation(
                x_i + K.dot(c_tm1 * rec_dp_mask[0], self.recurrent_kernel_i)
            )
            f = self.recurrent_activation(
                x_f + K.dot(c_tm1 * rec_dp_mask[1], self.recurrent_kernel_f)
            )
            c = f * c_tm1 + i * self.activation(
                x_c + K.dot(c_tm1 * rec_dp_mask[2], self.recurrent_kernel_c)
            )
            o = self.recurrent_activation(
                x_o + K.dot(c_tm1 * rec_dp_mask[3], self.recurrent_kernel_o)
            )
        h = o * c
        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h, c]

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Note: The layer has been tested with Keras 1.x
        Example:
        
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2 - Get the attention scores
            hidden = LSTM(64, return_sequences=True)(words)
            sentence, word_scores = Attention(return_attention=True)(hidden)
        """
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]

def load_data(folder_path="./data"):
	print ("Loading data...")
	x_path = os.path.join(folder_path,"train_x.csv")
	y_path = os.path.join(folder_path,"train_y.csv")
	t_path = os.path.join(folder_path,"test_x.csv")
	x_train = pd.read_csv(x_path).values[0:119017,1]
	y_train = pd.read_csv(y_path).values[0:119017,1]
	x_test  = pd.read_csv(t_path).values[:,1]

	jieba.set_dictionary('data/dict.txt.big')
	split_sentences = []
	# sentence_lengths = []

	train_length = len(x_train)
	for sentence in x_train:
		words = list(jieba.cut(sentence, cut_all=False))
		split_sentences.append(words)
	for sentence in x_test:
		words = list(jieba.cut(sentence, cut_all=False))
		split_sentences.append(words)
		# sentence_lengths.append(len(words))
	return split_sentences, train_length, y_train

	# with open(os.path.join(folder_path,"split_data.txt"),'w',encoding='utf-8') as output:
	# 	for sentence in x_train:
	# 		words = jieba.cut(sentence, cut_all=False)
	# 		output.write(' '.join(words) + '\n')

# def embedding(split_sentences):

# class Attention(Layer):
# 	def __init__(self, **kwargs):
# 		super(Attention, self).__init__(**kwargs)

# 	def build(self, input_shape):
# 		# Create a trainable weight variable for this layer.
# 		self.kernel = self.add_weight(	name='kernel',
#         								shape=(input_shape[1]*input_shape[2], input_shape[1]),
# 										initializer='uniform')
# 		self.bias = self.add_weight(name='bias',
# 									shape=(input_shape[1],),
# 									initializer='uniform')
# 		super(Attention, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		flatten_x = K.batch_flatten(x)
		weights = K.bias_add(K.dot(flatten_x, self.kernel), self.bias)
		softmax_w = K.softmax(weights, axis=-1)
		return K.sum(K.expand_dims(softmax_w,-1)*x, axis=-1)

	def compute_output_shape(self, input_shape):
		return (input_shape[0],input_shape[1])


if __name__ == '__main__':
	folder_path = "./data"
	save_path = "./models"
	split_sentences, train_length, labels = load_data()

	# Build Word2Vec model
	if not os.path.exists("w2v_model.wv"):
		print ("Building w2v_model...")
		w2v_model = Word2Vec(size=embedding_dim, min_count=3)
		w2v_model.build_vocab(split_sentences)
		w2v_model.train(split_sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
		w2v_model.wv.save("w2v_model.wv")
	wv = KeyedVectors.load("w2v_model.wv", mmap='r')
	# sentence_lengths.sort()
	# print (sentence_lengths[len(sentence_lengths)//4])
	# print (sentence_lengths[len(sentence_lengths)//4*2])
	# print (sentence_lengths[len(sentence_lengths)//4*3])

	# Build word2idx.json and idx2word.json
	if not (os.path.exists("word2idx.json") and os.path.exists("idx2word.json")):
		print ("Building word2idx.json and idx2word.json...")
		word2idx = {}
		idx2word = {}
		idx2word[PAD] = "<PAD>"
		# idx2word[1] = "<BOS>"
		# idx2word[2] = "<EOS>"
		idx2word[UNK] = "<UNK>"
		word2idx["PAD"] = PAD
		# word2idx["BOS"] = 1
		# word2idx["EOS"] = 2
		word2idx["UNK"] = UNK
		idx = 2
		for s in split_sentences:
			for w in s:
				if (w in wv.vocab) and (w not in word2idx):
					word2idx[w] = idx
					idx2word[idx] = w
					idx += 1
		with open('word2idx.json', 'w') as fp:
			json.dump(word2idx, fp)
		with open('idx2word.json', 'w') as fp:
			json.dump(idx2word, fp)
	else:
		word2idx = json.load(open("word2idx.json"))
		idx2word = json.load(open("idx2word.json"))

	vocabulary_size = len(word2idx)
	embedding_matrix = np.zeros((vocabulary_size, embedding_dim))
	embedding_matrix[UNK] = np.random.rand(embedding_dim)


	# Save embedding_matrix as npy file
	if not os.path.exists("embedding_matrix.npy"):
		for idx in range(3,vocabulary_size):
			embedding_matrix[idx] = wv[idx2word[idx]]
		np.save("embedding_matrix.npy",embedding_matrix)
	else:
		embedding_matrix = np.load("embedding_matrix.npy")

	# convert split_sentences into indexes
	split_sentences = split_sentences[0:train_length]
	if not os.path.exists("idx_sentences.npy"):
		idx_sentences = []
		for idx,s in enumerate(split_sentences):
			new_sentence = [] # 1 for <BOS>
			for w in s:
				if w in word2idx:
					new_sentence.append(word2idx[w])
				else:
					new_sentence.append(1) # 3 for <UNK>
			if len(new_sentence) > max_time_steps:
				new_sentence = new_sentence[0:max_time_steps]
				# sentence_lengths[i] = max_time_steps
				# new_sentence[max_time_steps-1] = 2 # 2 for <EOS>
			else:
				l = max_time_steps - len(new_sentence)
				for i in range(l):
					new_sentence.append(0)
			# else:
			# 	new_sentence.append(2)
			# 	for i in range(max_time_steps-len(new_sentence)):
			# 		new_sentence.append(0) # 0 for <PAD>
			idx_sentences.append(new_sentence)
		idx_sentences = np.array(idx_sentences)
		np.save("idx_sentences.npy", idx_sentences)
	else:
		idx_sentences = np.load("idx_sentences.npy")

	LSTM1 = Bidirectional(LSTM(embedding_dim, return_sequences=True, unit_forget_bias=False, dropout=dropout_rate, recurrent_dropout=dropout_rate), merge_mode='sum')
	LSTM2 = Bidirectional(LSTM(embedding_dim, return_sequences=True, unit_forget_bias=False, dropout=dropout_rate, recurrent_dropout=dropout_rate), merge_mode='sum')
	LSTM1.cell = LSTMPeepholeCell(units=embedding_dim, dropout=dropout_rate)
	LSTM2.cell = LSTMPeepholeCell(units=embedding_dim, dropout=dropout_rate)
    
    
	model = Sequential()
	model.add(Embedding(vocabulary_size, embedding_dim, input_length=max_time_steps, weights=[embedding_matrix], trainable=False))
# 	model.add(Bidirectional(LSTM(embedding_dim, return_sequences=True, unit_forget_bias=False, dropout=dropout_rate, recurrent_dropout=dropout_rate), merge_mode='sum'))
# 	model.add(Bidirectional(LSTM(embedding_dim, return_sequences=True, unit_forget_bias=False, dropout=dropout_rate, recurrent_dropout=dropout_rate), merge_mode='sum'))
	model.add(LSTM1)
	model.add(LSTM2)
	model.add(Attention())
	model.add(Dense(256))
	model.add(LeakyReLU(0.2))
	# model.add(Dense(256))
	# model.add(LeakyReLU(0.2))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	opt = optimizers.Adam(lr=0.001)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()
		
	checkpoint = ModelCheckpoint(os.path.join(save_path,"model_best.h5"), monitor='val_acc', save_best_only=True)

	model.fit(idx_sentences, np.array(labels), validation_split=0.2, shuffle=True, callbacks=[checkpoint], epochs=epochs, batch_size=batch_size)



	# print (wv.most_similar(positive=["魯蛇"]))
	# embedded_sentences = [ [wv[word] for word in sentence] for sentence in split_sentences ]