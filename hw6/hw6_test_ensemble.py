import pandas as pd
import numpy as np
import os
import sys
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
from attention import Attention
from keras import backend as K

max_time_steps = 100
embedding_dim = 100
epochs = 100
batch_size = 1000

PAD = 0
UNK = 1


def load_data(test_x_path, dict_path):
	x_path = test_x_path
	# y_path = os.path.join(folder_path,"train_y.csv")
	x_test = pd.read_csv(x_path).values[:,1]
	# y_train = pd.read_csv(y_path).values[:,1]

	jieba.set_dictionary(dict_path)
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
	test_x_path = sys.argv[1]
	dict_path = sys.argv[2]
	output_path = sys.argv[3]
	split_sentences, sentence_lengths = load_data(test_x_path, dict_path)

	if not os.path.exists("word2idx.json"):
		print("json files not exist!")
	word2idx = json.load(open("word2idx.json"))

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
				new_sentence.append(UNK) # 3 for <UNK>
		if len(new_sentence) > max_time_steps:
			new_sentence = new_sentence[0:max_time_steps]
			sentence_lengths[i] = max_time_steps
			# new_sentence[max_time_steps-1] = 2 # 2 for <EOS>
		else:
			l = max_time_steps - len(new_sentence)
			for i in range(l):
				new_sentence.append(PAD)
		# else:
		# 	new_sentence.append(2)
		# 	for i in range(max_time_steps-len(new_sentence)):
		# 		new_sentence.append(0) # 0 for <PAD>
		idx_sentences.append(new_sentence)
	idx_sentences = np.array(idx_sentences)

	model1 = load_model("model_5.h5", custom_objects={'Attention': Attention()})
	model2 = load_model("model_6.h5", custom_objects={'Attention': Attention()})
	model3 = load_model("model_7.h5", custom_objects={'Attention': Attention()})
	model4 = load_model("model_4.h5", custom_objects={'Attention': Attention()})

	iterations = idx_sentences.shape[0] // batch_size

	output_list = [["id", "label"]]

	K.set_learning_phase(0) # Set predict mode
	sess = K.get_session()

	output1 = model1.layers[-2].output
	output2 = model2.layers[-2].output
	output3 = model3.layers[-2].output
	output4 = model4.layers[-2].output
	predictions = K.sigmoid((output1+output2+output3+output4)/4.0)

	for i in range(iterations):
		test_x = idx_sentences[i*batch_size:(i+1)*batch_size]
		preds = sess.run(predictions, feed_dict={model1.input: test_x, model2.input: test_x, model3.input: test_x, model4.input: test_x})
		# preds1 = model1.predict(test_x)
		# preds2 = model2.predict(test_x)
		for j in range(batch_size):
			output_list = output_list + [[i*batch_size+j, int(round(preds[j][0]))]]
	# print (output_list)
	output_file = pd.DataFrame(output_list)
	output_file.to_csv(output_path, index=False, header=False)
	print ("Generate ans.csv!")
	


	# print (wv.most_similar(positive=["魯蛇"]))
	# embedded_sentences = [ [wv[word] for word in sentence] for sentence in split_sentences ]