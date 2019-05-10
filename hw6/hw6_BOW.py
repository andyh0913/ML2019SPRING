# train BOW

import pandas as pd
import numpy as np
import os
import random
import jieba
from gensim.models import Word2Vec
from gensim import models
from gensim.models import KeyedVectors
import json
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Bidirectional, LeakyReLU, multiply, Layer, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import optimizers
from keras import regularizers

# import heapq

max_time_steps = 100
embedding_dim = 100
epochs = 200
batch_size = 500
valid_size = 19017

dropout_rate = 0.5

# Constants
PAD = 0
UNK = 1

def load_data(folder_path="/content/drive/My Drive/ML2019Spring/hw6/data"):
    print ("Loading data...")
    x_path = os.path.join(folder_path,"train_x.csv")
    y_path = os.path.join(folder_path,"train_y.csv")
    t_path = os.path.join(folder_path,"test_x.csv")
    x_train = pd.read_csv(x_path).values[0:119017,1]
    y_train = pd.read_csv(y_path).values[0:119017,1]
    x_test  = pd.read_csv(t_path).values[:,1]

    jieba.set_dictionary('/content/drive/My Drive/ML2019Spring/hw6/data/dict.txt.big')
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

def data_generator(train_x, train_y, batch_size, word_dict_size, shuffle=True):
    train_size = len(train_x)
    print (train_size)
    iterations = train_size // batch_size
    while True:
        if shuffle:
            mapIndexPosition = list(zip(train_x, train_y))
            random.shuffle(mapIndexPosition)
            train_x, train_y = zip(*mapIndexPosition)
            p = np.random.permutation(train_size)
        for i in range(iterations):
            batch_bow_sentences = np.zeros((batch_size,word_dict_size))
            for j in range(i*batch_size, (i+1)*batch_size):
                for k in range(len(train_x[j])):
                    if train_x[j][k] in word_dict:
                        batch_bow_sentences[j%batch_size][ word_dict[train_x[j][k]] ] += 1
                    else:
                        batch_bow_sentences[j%batch_size][0] += 1
#             print ("Feed batched train data!")
            batch_bow_sentences = batch_bow_sentences / np.max(batch_bow_sentences, axis=-1, keepdims=True)
            yield (batch_bow_sentences, np.array(train_y[i*batch_size:(i+1)*batch_size]))
            
def valid_generator(valid_x, valid_y, batch_size, word_dict_size):
    iterations = valid_size // batch_size
    while True:
        for i in range(iterations):
            batch_bow_sentences = np.zeros((batch_size,word_dict_size))
            for j in range(i*batch_size, (i+1)*batch_size):
                for k in range(len(valid_x[j])):
                    if valid_x[j][k] in word_dict:
                        batch_bow_sentences[j%batch_size][ word_dict[valid_x[j][k]] ] += 1
                    else:
                        batch_bow_sentences[j%batch_size][0] += 1
#             print ("Feed batched valid data!")
            yield (batch_bow_sentences, np.array(valid_y[i*batch_size:(i+1)*batch_size]))
    

if __name__ == '__main__':
    folder_path = "./data"
    save_path = "./models"
    split_sentences, train_length, labels = load_data()

    # Build word dictionary
    if not os.path.exists("/content/drive/My Drive/ML2019Spring/hw6/word_dict.json"):
        print("Building word_dict.json...")
        word_dict = {}
        word_count = {}
        word_dict['<UNK>'] = 0
        idx = 1
        for s in split_sentences:
            for w in s:
                if not w in word_count:
                    word_count[w] = 1
                elif word_count[w] < 2:
                    word_count[w] += 1
                elif not w in word_dict:
                    word_dict[w] = idx
                    idx += 1
        with open('/content/drive/My Drive/ML2019Spring/hw6/word_dict.json', 'w') as fp:
            json.dump(word_dict, fp)
    else:
        word_dict = json.load(open("/content/drive/My Drive/ML2019Spring/hw6/word_dict.json"))
    word_dict_size = len(word_dict)
    print ("Size of word_dict.json is:",word_dict_size)
    split_sentences = split_sentences[0:train_length]

    # convert split_sentences into indexes
    train_x = split_sentences[0:100000]
    train_y = labels[0:100000]
    valid_x = split_sentences[100000:119017]
    valid_y = labels[100000:119017]
#     if not os.path.exists("/content/drive/My Drive/ML2019Spring/hw6/valid_data.npy"):
#         print ("Building valid_data...")
#         bow_valid_x = np.zeros((valid_size, word_dict_size))

#         for idx,s in enumerate(valid_x):
#             for w in s:
#                 bow_valid_x[idx][word_dict[w]] += 1

#         np.save("/content/drive/My Drive/ML2019Spring/hw6/bow_sentences.npy", idx_sentences)
#     else:
#         bow_valid_x = np.load("/content/drive/My Drive/ML2019Spring/hw6/bow_sentences.npy")

    model = Sequential()
#     model.add(BatchNormalization())
#     model.add(Dense(50, input_shape=(word_dict_size, )))
    model.add(Dense(1024, input_shape=(word_dict_size, ), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
#     model.add(Dropout(dropout_rate))
#     model.add(Dense(512, kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
    model.add(Dense(512))  
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
#     model.add(Dropout(dropout_rate))
#     model.add(Dense(256, kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
#     model.add(Dropout(dropout_rate))
#     model.add(Dense(128, kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
    model.add(Dense(128))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
#     model.add(Dropout(dropout_rate))
#     model.add(Dense(64, kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
    model.add(Dense(64))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
#     model.add(Dropout(dropout_rate))
#     model.add(Dense(1, kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    opt = optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    checkpoint = ModelCheckpoint(os.path.join(save_path,"/content/drive/My Drive/ML2019Spring/hw6/model_bow_best.h5"), monitor='val_acc', save_best_only=True)

    # model.fit(idx_sentences, np.array(labels), validation_split=0.2, shuffle=True, callbacks=[checkpoint], epochs=epochs, batch_size=batch_size)
    model.fit_generator(data_generator(train_x, train_y, batch_size=batch_size, word_dict_size=word_dict_size), 
                        validation_data=valid_generator(valid_x, valid_y, batch_size, word_dict_size), validation_steps=valid_size//batch_size, 
                        steps_per_epoch=len(train_x)//batch_size, epochs=epochs)


    # print (wv.most_similar(positive=["魯蛇"]))
    # embedded_sentences = [ [wv[word] for word in sentence] for sentence in split_sentences ]