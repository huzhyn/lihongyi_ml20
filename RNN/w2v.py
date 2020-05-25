# -*- coding: utf-8 -*-

# w2v.py
# 這個 block 是用來訓練 word to vector 的 word embedding
# 注意！這個 block 在訓練 word to vector 時是用 cpu，可能要花到 10 分鐘以上
import os
from utils import *
from gensim.models import word2vec


def train_word2vec(x):
    # 訓練 word to vector 的 word embedding
    model = word2vec.Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=1)
    return model

# 利用成熟的gensim.models.word2vec 训练自己的word vector
if __name__ == "__main__":
    prefix = os.getcwd()
    training_with_label = os.path.join(prefix, "./data/training_label.txt")
    training_with_nolabel = os.path.join(prefix, "./data/training_nolabel.txt")

    testing_data = os.path.join(prefix, "./data/testing_data.txt")

    print("loading training data ...")
    train_x, y = load_training_data(training_with_label)
    train_x_no_label = load_training_data(training_with_nolabel)

    print("loading testing data ...")
    test_x = load_testing_data(testing_data)

    # model = train_word2vec(train_x + train_x_no_label + test_x)
    model = train_word2vec(train_x + test_x)

    print("saving model ...")
    # model.save(os.path.join(path_prefix, 'model/w2v_all.model'))
    model.save(os.path.join(prefix, 'w2v_all.model'))