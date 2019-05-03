import pandas as pd
import numpy as np
import os
import jieba
from gensim.models import Word2Vec
from gensim import models
from gensim.models import KeyedVectors
import json
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint

from heapq import nlargest

max_time_steps = 100
embedding_dim = 100
epochs = 100
batch_size = 1000


def load_data(folder_path="./data"):
	x_path = os.path.join(folder_path,"train_x.csv")
	y_path = os.path.join(folder_path,"train_y.csv")
	t_path = os.path.join(folder_path,"test_x.csv")
	x_train = pd.read_csv(x_path).values[:,1]
	y_train = pd.read_csv(y_path).values[:,1]
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

if __name__ == '__main__':
	folder_path = "./data"
	save_path = "./models"
	split_sentences, train_length, labels = load_data()

	# Build Word2Vec model
	if not os.path.exists("w2v_model.wv"):
		print ("Building w2v_model...")
		w2v_model = Word2Vec(size=100,min_count=3)
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
		word2idx = {}
		idx2word = {}
		idx2word[0] = "<PAD>"
		# idx2word[1] = "<BOS>"
		# idx2word[2] = "<EOS>"
		idx2word[1] = "<UNK>"
		word2idx["PAD"] = 0
		# word2idx["BOS"] = 1
		# word2idx["EOS"] = 2
		word2idx["UNK"] = 1
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
				sentence_lengths[i] = max_time_steps
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

	model = Sequential()
	model.add(Embedding(vocabulary_size, embedding_dim, input_length=max_time_steps, weights=[embedding_matrix], trainable=False))
	model.add(LSTM(embedding_dim, return_sequences=True))
	model.add(LSTM(embedding_dim))
	model.add(Dense(256))
	model.add(Dropout(0.2))
	model.add(Dense(256))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	
	checkpoint = ModelCheckpoint(os.path.join(save_path,"model_best.h5"), monitor='val_acc', save_best_only=True)

	model.fit(idx_sentences, np.array(labels), validation_split=0.2, shuffle=True, callbacks=[checkpoint], epochs=epochs, batch_size=batch_size)



	# print (wv.most_similar(positive=["魯蛇"]))
	# embedded_sentences = [ [wv[word] for word in sentence] for sentence in split_sentences ]