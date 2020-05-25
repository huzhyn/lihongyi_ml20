# -*- coding: utf-8 -*-
# preprocess.py
# 這個 block 用來做 data 的預處理

import torch
import os
from gensim.models import Word2Vec
from utils import load_training_data, load_testing_data

# 注意这里类的使用
class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path="./w2v.model"):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []  # index -> word，列表就行，因为位置就代表了index
        self.word2idx = {}  # word -> index
        self.embedding_matrix = [] # word embedding, 列表的位置信息代表了index，每个元素是实际的vector

    def get_w2v_model(self):
        # 把之前訓練好的 word to vec 模型讀進來
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def add_embedding(self, word):
        # 把 word 加進 embedding，並賦予他一個隨機生成的 representation vector
        # word 只會是 "<PAD>" 或 "<UNK>"
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self, load=True):
        print("Get embedding ...")
        # 取得訓練好的 Word2vec word embedding
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # 製作一個 word2idx 的 dictionary
        # 製作一個 idx2word 的 list
        # 製作一個 word2vector 的 list
        for i, word in enumerate(self.embedding.wv.vocab):
            print('get words #{}'.format(i+1), end='\r')
            #e.g. self.word2index['he'] = 1
            #e.g. self.index2word[1] = 'he'
            #e.g. self.vectors[1] = 'he' vector
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        print('')
        self.embedding_matrix = torch.tensor(self.embedding_matrix)  # list -> tensor
        # 將 "<PAD>" 跟 "<UNK>" 加進 embedding 裡面
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        # 將每個句子變成一樣的長度
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self):
        # 把句子裡面的字轉成相對應的 index
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 將每個句子變成一樣的長度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)

    def labels_to_tensor(self, y):
        # 把 labels 轉成 tensor
        y = [int(label) for label in y]
        return torch.LongTensor(y)

# 这个数据的处理方法应该是最值得借鉴的了
# 注意这里padding的方法，以后也可以使用，指定长度，短的补长的截，简单方便
# index和word的关系，以及存储方法，也值得借鉴，之前在处理同义词时已经想到这个方法了
if __name__ == "__main__":
    prefix = os.getcwd()
    training_with_label = os.path.join(prefix, "./data/training_label.txt")
    training_with_nolabel = os.path.join(prefix, "./data/training_nolabel.txt")

    testing_data = os.path.join(prefix, "./data/testing_data.txt")
    w2v_path = os.path.join(prefix, "w2v_all.model")

    print("loading training data ...")
    train_x, y = load_training_data(training_with_label)
    train_x_no_label = load_training_data(training_with_nolabel)

    print("loading testing data ...")
    test_x = load_testing_data(testing_data)

    sen_len = 20
    preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x = preprocess.sentence_word2idx()
    y = preprocess.labels_to_tensor(y)