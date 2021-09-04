import pandas as pd
import GetBowFeatures
import numpy as np
import os
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tensorflow.keras.models import load_model
import xgboost
from Bio import SeqIO
import warnings
import argparse
warnings.filterwarnings("ignore")


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


def get_cbow_features(content, model):
    '''
    extract the CBOW features
    '''
    line = content.split()
    single_feature = np.mean(model[line], axis=0, keepdims=True)
    return single_feature


def get_combined_features(seq):
    features = GetBowFeatures.get_single_features(seq)
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
    models_name = [model_1, model_2, model_3]
    for i in range(2, 5):
        text = ""
        for k in range(len(seq) - i + 1):
            if k != len(seq) - i:
                text = text + seq[k: k + i] + " "
            else:
                text = text + seq[k: k + i]
        cbow_feature = get_cbow_features(text, models_name[i - 2])
        features = np.c_[features, cbow_feature]
    return features


def prediction_result(features):
    ann_model = load_model("../files/DNN.h5", compile=False)
    xgb = xgboost.XGBClassifier()
    xgb.load_model('../files/xgboost.model')
    final_features = ann_model.predict(features)
    prediction = xgb.predict(final_features)
    if prediction.flatten() == 1:
        return "GPCR"
    else:
        return "Non-GPCR"


def file_prediction_results(gpcr_save_path):
    columns_name = ['protein_name', 'protein_sequence']
    data = pd.DataFrame(columns=columns_name)
    data['prediction'] = None
    for seq_gpcr in SeqIO.parse(gpcr_save_path, 'fasta'):
        data = data.append([{'protein_name': seq_gpcr.name, 'protein_sequence': str(seq_gpcr.seq)}], ignore_index=True)
    for index in range(len(data)):
        gpcr_sequence = data.iloc[index, 1]
        try:
            gpcr_features = get_combined_features(gpcr_sequence)
        except Exception as e:
            continue
        data.iloc[index, 2] = prediction_result(gpcr_features)
    results_path = gpcr_save_path.replace("protein", "results")
    results_path = results_path.replace(".fasta", ".csv")
    data.to_csv(results_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict gpcr')
    parser.add_argument("path", help="the fasta file path")
    args = parser.parse_args()
    file_prediction_results(args.path)