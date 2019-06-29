import gensim
from gensim import corpora, models, similarities
import os
import jieba
import pandas as pd
import numpy as np
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from collections import OrderedDict
import json

news_path = "../newsdata/mod_dict"
phrase_path = "../gensim_corpora/mod_dict_phrase"
stopword_path = "../dictionary/stopword.txt"
added_query_path = "../tmp/added_query_2.txt"
jieba.set_dictionary("../dictionary/dict_mod.txt.big")
QSf = pd.read_csv("../dataset/QS_1.csv")
TDf = pd.read_csv("../dataset/TD.csv")

EXPAND_SIZE = 50
output_filename = "../output/300_addtraining_noneg_tfidf_expand_top{}.csv".format(str(EXPAND_SIZE))

'''
new_query = origin_weight * query + add_query_weight * add_query + train_weight * train_query + expand_weight * expand_query
expand_word_num is how much word added from expansion files
'''
ORIGIN_WEIGHT = 2
ADD_QUERY_WEIGHT = 32
TRAIN_WEIGHT = 8
EXPAND_WEIGHT = 2
TRAIN_EXPAND_WORD_NUM = 100
FEEDBACK_EXPAND_WORD_NUM = 300


class Retriever:

    #self.texts = list()
    #self.index2fname = list()
    #self.fname2index = dict()
    #self.dictionary = None
    #self.corpus = None
    #self.tfidf = None
    #self.tfidf_corpus = None
    #self.query_list = list()
    #self.query_tfidf = None
    #self.index = None
    #self.sims = None

    def __init__(self):
        file_names = sorted(os.listdir(news_path))
        self.__read_news__(file_names)
        self.__build_dict__()
        self.__build_tfidf__()
        self.__get_querys__()
        self.__get_corpus_index__()
        self.__expand_query_1__()
        self.__expand_query_2__()
        self.__tfidf_predict__()
        self.__expand_query_3__()
        self.__tfidf_predict__()
        self.__finetune__()
        self.__generate_result__()

    def __read_news__(self, file_names):
        print("Read_news")
        self.texts = list()
        self.index2fname = list()
        self.fname2index = dict()
        index = 0
        with open(stopword_path, 'r') as f:
            self.stopwords = set(f.read().split('\n'))
        for fname in file_names:
            if not fname.startswith("news_"):
                continue
            with open(os.path.join(news_path, fname), encoding="utf-8") as f:
                text = f.read().split(' ')
                text = self.__cut_string__(text)
                self.texts.append(text)
                self.index2fname.append(fname)
                self.fname2index[fname] = index
                index += 1
    
    def __build_dict__(self):
        print("Build_dict")
        # Building phrase dict and corpus
        dict_path = os.path.join(phrase_path, "news.dict")
        mm_path = os.path.join(phrase_path, "news.mm")

        if os.path.exists(dict_path):
            self.dictionary = corpora.Dictionary.load(dict_path)
            self.corpus = corpora.MmCorpus(mm_path)
        else:
            bigram = Phrases(self.texts, min_count=1, threshold=2, delimiter=b' ')
            bigram_phraser = Phraser(bigram)
            bigram_token = list()
            for sent in self.texts:
                bigram_token.append(bigram_phraser[sent])
            self.dictionary = corpora.Dictionary(bigram_token)
            self.dictionary.save(dict_path)
            self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
            corpora.MmCorpus.serialize(mm_path, self.corpus)

    def __build_tfidf__(self):
        print("Build_tfidf")
        # build tfidf model
        tfidf_path = os.path.join(phrase_path, "model.tfidf")
        if os.path.exists(tfidf_path):
            self.tfidf = models.TfidfModel.load(tfidf_path)
        else:
            pivot = 0
            for bow in self.corpus:
                pivot += len(bow)
            pivot /= len(self.corpus)
            self.tfidf = models.TfidfModel(self.corpus, id2word=self.dictionary, smartirs="Ltc", pivot=pivot, slope=0.2)
            self.tfidf.save(tfidf_path)
        self.tfidf_corpus = self.tfidf[self.corpus]

    def __get_querys__(self):
        print("Get_query")
        raw_querys = list(QSf["Query"])
        self.query_list = list()
        for q in raw_querys:
            text = jieba.cut(q, cut_all=False)
            self.query_list.append(self.__cut_string__(text))
    
    def __expand_query_1__(self):
        print("Expand_1")
        # expand query from user defined data and generate query_tfidf
        with open(added_query_path, "r") as f:
            expand_docs = f.read().split("\n")
            for i, expand_words in enumerate(expand_docs):
                self.query_list[i] += ADD_QUERY_WEIGHT * expand_words.split(" ")
        query_vecs = list()
        for q in self.query_list:
            query_vecs.append(self.dictionary.doc2bow(q))
        tmp_query_tf_idf = self.tfidf[query_vecs]
        self.query_tfidf = list()
        for query in tmp_query_tf_idf:
            self.query_tfidf.append(self.__vec2tfidf__(self.__tfidf2vec__(query)))
    def __expand_from_files__(self, query_idx, expand_index, state):
        word_dict = dict()
        q = self.query_tfidf[query_idx]
        print("original query", q)
        if state == 2:
            w = TRAIN_WEIGHT
            ex = TRAIN_EXPAND_WORD_NUM
        elif state == 3:
            w = EXPAND_WEIGHT
            ex = FEEDBACK_EXPAND_WORD_NUM

        # extract all word from expand docs
        expand_vecs = []
        for idx in expand_index:
            vec = self.__tfidf2vec__(self.tfidf_corpus[idx])
            expand_vecs.append(vec)
        expand_vecs = np.array(expand_vecs)

        # extract top K words
        expand_vector = expand_vecs.mean(axis=0)
        idxs_to_delete = np.argsort(expand_vector)[::-1][ex:]
        expand_vector[idxs_to_delete] = 0

        # calculate new query
        q_vec = self.__tfidf2vec__(q)
        q = (ORIGIN_WEIGHT * q_vec + w * expand_vector) / (ORIGIN_WEIGHT + w)
        self.query_tfidf[query_idx] = self.__vec2tfidf__(q)

        print("new query", self.query_tfidf[query_idx])

    def __expand_query_2__(self):
        print("Expand_2")
        # expand query from training data
        train_query = list(TDf["Query"])
        train_fname = list(TDf["News_Index"])
        train_relevance = list(TDf["Relevance"])
        test_query = list(QSf["Query"])
        print(test_query)

        expand_index = list()
        prev_query = train_query[0]
        for i in range(len(train_query)):
            q = train_query[i]
            if prev_query != q and len(expand_index) != 0:
                print("expanding {}".format(prev_query))
                self.__expand_from_files__(query_idx, expand_index, 2)
                expand_index.clear()
            if q in test_query and int(train_relevance[i]) == 3:
                query_idx = test_query.index(q)
                expand_index.append(self.fname2index[train_fname[i]])
            elif q.startswith("堅決反對政府") and int(train_relevance[i]) == 0:
                query_idx = 12
                expand_index.append(self.fname2index[train_fname[i]])
            prev_query = q

    def __tfidf2vec__(self, tfidf):
        new_vec = np.zeros(len(self.dictionary))
        for tup in tfidf:
            new_vec[tup[0]] = tup[1]
        return new_vec

    def __vec2tfidf__(self, vec):
        new_tfidf = []
        assert(vec.shape[0] == len(self.dictionary))
        for i in range(len(self.dictionary)):
            if vec[i] != 0:
                new_tfidf.append((i, vec[i]))
        return new_tfidf

    def __tfidf_average__(self, tfidf_list):
        vec_list = np.zeros((len(tfidf_list), len(self.dictionary)))
        for i, tfidf in enumerate(tfidf_list):
            vec_list[i] = self.__tfidf2vec__(tfidf)
        tfidf_mean = self.__vec2tfidf__(vec_list.mean(axis = 0))
        return tfidf_mean

    def __expand_query_3__(self):
        print("Expand_3")
        # expand query from EXPAND_SIZE
        

        '''k = 20
        corpus = np.array(self.corpus)
        top_k = list()
        for s in self.sims:
            top_k.append(np.argsort(s)[-k:][::-1])
        new_querys_tfidf = []
        for arr in top_k:
            tfidf_list = self.tfidf[corpus[arr]]
            tfidf_mean = self.__tfidf_average__(tfidf_list)
            new_querys_tfidf.append(tfidf_mean)
        self.query_tfidf = new_querys_tfidf'''
        k = EXPAND_SIZE
        top_k = list()
        for s in self.sims:
            top_k.append(np.argsort(s)[-k:][::-1])
        for query_idx, arr in enumerate(top_k):
            self.__expand_from_files__(query_idx, list(arr), 3)

    def __get_corpus_index__(self):
        print("Get_corpus_index")
        index_path = os.path.join(phrase_path, "model.index")
        if os.path.exists(index_path):
            self.index = similarities.MatrixSimilarity.load(index_path)
        else:
            featureNum = len(self.dictionary.token2id.keys())
            self.index = similarities.SparseMatrixSimilarity(self.tfidf_corpus, num_features=featureNum)
            self.index.save(index_path)

    def __tfidf_predict__(self):
        print("Predict")
        self.sims = self.index[self.query_tfidf]
        self.sims = np.array(self.sims)

    def __finetune__(self):
        print("Finetune")
        # fine tune training data
        querys_in_td = { 12:"堅決反對政府舉債發展前瞻建設計畫", 15:"支持陳前總統保外就醫", 
        16:"年金改革應取消或應調降軍公教月退之優存利率十八趴", 17:"同意動物實驗",
        18:"油價應該凍漲或緩漲", 19:"反對旺旺中時併購中嘉" }
        delete = 0
        best = 3
        nice = 2
        for key, query in querys_in_td.items():
            best_count = 0
            nice_count = 0
            best_sum = 0
            nice_sum = 0
            td_idx_list = TDf[TDf['Query'] == query].index.tolist()
            if key == 12:
                delete = 3
                best = 0
                nice = 1
                good = 2
            else: 
                delete = 0
                best = 3
                nice = 2
                good = 1
            for td_idx in td_idx_list:
                if TDf['Relevance'][td_idx] == delete:
                    # idx_to_delete.append(self.fname2index[TDf['News_Index'][td_idx]])
                    self.sims[key][self.fname2index[TDf['News_Index'][td_idx]]] = -1
                elif TDf['Relevance'][td_idx] == best:
                    best_sum += self.sims[key][self.fname2index[TDf['News_Index'][td_idx]]]
                    if key != 12:
                        self.sims[key][self.fname2index[TDf['News_Index'][td_idx]]] += 3
                    best_count += 1
                elif TDf['Relevance'][td_idx] == nice:
                    if key != 12:
                        self.sims[key][self.fname2index[TDf['News_Index'][td_idx]]] += 2
                    nice_sum += self.sims[key][self.fname2index[TDf['News_Index'][td_idx]]]
                    nice_count += 1
                elif TDf['Relevance'][td_idx] == good:
                    if key != 12:
                        self.sims[key][self.fname2index[TDf['News_Index'][td_idx]]] += 1
            if key != 18:
                best_sum /= best_count
                nice_sum /= nice_count
                print(query, 'top_300_avg', (self.sims[key][np.argsort(self.sims[key])[-300:][::-1]]).sum() / 300,'best_avg: ', best_sum, 'nice_avg: ', nice_sum)

        # fine tune: ours
        json_list = list()
        # with open('1To6.json', 'r') as f:
        #     data = json.load(f)
        #     json_list.append(data)
        with open('1To3.json', 'r') as f1:
            data = json.load(f1)
            json_list.append(data)
        with open('7To8.json', 'r') as f2:
            data = json.load(f2)
            json_list.append(data)
        with open('9To14.json', 'r') as f3:
            data = json.load(f3)
            json_list.append(data)
        for js in json_list:
            for key, news_pairs in js.items():
                query_id = int(key)-1
                for news_name, label in news_pairs.items():
                    if label == '0':
                        self.sims[query_id][self.fname2index[news_name]] = -1
        print(self.sims)

    def __generate_result__(self):
        top = 300
        top_n = list()
        Top_300 = list()
        for s in self.sims:
            top_n.append(np.argsort(s)[-top:][::-1])
        for arr in top_n:
            arr_list = list()
            for j in range(top):
                arr_list.append(self.index2fname[arr[j]][-11:])
            Top_300.append(np.array(arr_list))
        df = pd.DataFrame(Top_300)
        df.columns = ['Rank_'+str(i).zfill(3) for i in range(1, top+1)]
        df.index = ['q_'+str(i).zfill(2) for i in range(1, 21)]
        df.index.name = 'Query_Index'
        df.to_csv(output_filename, index='True', sep=',')

    def __cut_string__(self, text):
        text = list(text)
        length = len(text)
        i = 0
        while i < length:
            if text[i] == '' or text[i] == '\n' or text[i] in self.stopwords:
                del text[i]     
                i -= 1
                length -= 1
            i += 1
        return text


if __name__ == "__main__":
    Retriever()