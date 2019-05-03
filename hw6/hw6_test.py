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
from keras.models import load_model

max_time_steps = 100
embedding_dim = 100
epochs = 100
batch_size = 1000


def load_data(folder_path="./data"):
	x_path = os.path.join(folder_path,"test_x.csv")
	# y_path = os.path.join(folder_path,"train_y.csv")
	x_test = pd.read_csv(x_path).values[:,1]
	# y_train = pd.read_csv(y_path).values[:,1]

	jieba.set_dictionary('data/dict.txt.big')
	split_sentences = []
	sentence_lengths = []

	for sentence in x_test:
		words = list(jieba.cut(sentence, cut_all=False))
		split_sentences.append(words)
		sentence_lengths.append(len(words))
	return split_sentences, sentence_lengths

	# with open(os.path.join(folder_path,"split_data.txt"),'w',encoding='utf-8') as output:
	# 	for sentence in x_train:
	# 		words = jieba.cut(sentence, cut_all=False)
	# 		output.write(' '.join(words) + '\n')

# def embedding(split_sentences):

if __name__ == '__main__':
	folder_path = "./data"
	save_path = "./models"
	output_path = "./result/ans.csv"
	split_sentences, sentence_lengths = load_data()
	if not os.path.exists("w2v_model.wv"):
		print ("w2v_model.wv not exists!")
	wv = KeyedVectors.load("w2v_model.wv", mmap='r')

	if not (os.path.exists("word2idx.json") and os.path.exists("idx2word.json")):
		print("json files not exist!")
	word2idx = json.load(open("word2idx.json"))
	idx2word = json.load(open("idx2word.json"))

	vocabulary_size = len(word2idx)
	# embedding_matrix = np.zeros((vocabulary_size, embedding_dim))

	# if not os.path.exists("embedding_matrix.npy"):
	# 	for idx in range(3,vocabulary_size):
	# 		embedding_matrix[idx] = wv[idx2word[idx]]
	# 	np.save("embedding_matrix.npy",embedding_matrix)
	# else:
	# 	embedding_matrix = np.load("embedding_matrix.npy")

	# convert split_sentences into indexes
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

	model = load_model("models/model_best.h5")
	iterations = idx_sentences.shape[0] // batch_size

	output_list = [["id", "label"]]

	for i in range(iterations):
		test_x = idx_sentences[i*batch_size:(i+1)*batch_size]
		preds = model.predict(test_x)
		for j in range(batch_size):
			output_list = output_list + [[i*batch_size+j, int(round(preds[j][0]))]]
	print (output_list)
	output_file = pd.DataFrame(output_list)
	output_file.to_csv(output_path, index=False, header=False)
	print ("Generate ans.csv!")
	


	# print (wv.most_similar(positive=["魯蛇"]))
	# embedded_sentences = [ [wv[word] for word in sentence] for sentence in split_sentences ]