from gensim.models import word2vec, Word2Vec
import numpy as np
from gensim.models.callbacks import CallbackAny2Vec
import os
from tqdm import tqdm
import totxt


def train_model():
    '''
    train the CBOW models
    '''
    print("train cbow models...")
    totxt.get_text(r'../files/all_data.xlsx')
    print("train cbow models 1 .. Please wait patiently")
    sentences = word2vec.Text8Corpus(r'..\files\all_2.txt')
    word2vec.Word2Vec(sentences, min_count=0, size=128, window=5, compute_loss=True, iter=10, workers=8, callbacks=[EpochLogger("l2")])
    print("train cbow models 2 .. Please wait patiently")
    sentences = word2vec.Text8Corpus(r'..\files\all_3.txt')
    word2vec.Word2Vec(sentences, min_count=0, size=256, window=5, compute_loss=True, iter=10, workers=8, callbacks=[EpochLogger("l3")])
    print("train cbow models 3 .. Please wait patiently")
    sentences = word2vec.Text8Corpus(r'..\files\all_4.txt')
    word2vec.Word2Vec(sentences, min_count=0, size=512, window=5, compute_loss=True, iter=10, workers=8, callbacks=[EpochLogger("l4")])


def get_features(file_path, model):
    '''
    extract the CBOW features
    '''
    f = open(file_path, 'r')
    content = f.readlines()
    f.close()
    count = 0
    features = None
    for line in tqdm(content):
        line = line.split()
        single_feature = np.mean(model[line], axis=0, keepdims=True)
        if count == 0:
            features = single_feature
            count += 1
        else:
            features = np.r_[features, single_feature]
    # np.save(r'../files/Liao/models/features_4.npy', features)
    return features


def merge_features():
    '''
    concatenate the three kinds CBOW features
    '''
    train_model()
    print("extract cbow features...")
    if os.path.exists(r'../files/l2.model'):
        model_1 = Word2Vec.load(r'../files/l2.model')
    else:
        model_1 = Word2Vec.load(r'../files/last_l2.model')
    if os.path.exists(r'../files/l3.model'):
        model_2 = Word2Vec.load(r'../files/l3.model')
    else:
        model_2 = Word2Vec.load(r'../files/last_l3.model')
    if os.path.exists(r'../files/l4.model'):
        model_3 = Word2Vec.load(r'../files/l4.model')
    else:
        model_3 = Word2Vec.load(r'../files/last_l4.model')
    feautures_1 = get_features(r'../files/all_2.txt', model_1)
    feautures_2 = get_features(r'../files/all_3.txt', model_2)
    feautures_3 = get_features(r'../files/all_4.txt', model_3)
    CBOW_features = np.concatenate([feautures_1, feautures_2, feautures_3], axis=1)
    np.save(r'../files/CBOW_features.npy', CBOW_features)


class EpochLogger(CallbackAny2Vec):
    def __init__(self, name):
        self.epoch = 1
        self.losses = []
        self.previous_losses = 0
        self.add_losses = []
        self.name = name

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        self.epoch += 1
        self.add_losses.append(loss - self.previous_losses)
        self.previous_losses = loss
        if self.epoch > 2:
            if self.add_losses[-1] == 0 and self.add_losses[-2] != 0:
                path = r"../files" + os.sep + "{}.model".format(self.name)
                model.save(path)

    def on_train_end(self, model):
        if self.add_losses[-1] == 0:
            pass
        else:
            path = r"../files" + os.sep + "last_{}.model".format(self.name)
            model.save(path)


if __name__ == '__main__':
    merge_features()