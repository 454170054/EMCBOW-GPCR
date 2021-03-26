import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import glorot_uniform
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import numpy as np
from tqdm import tqdm
import GetBowFeatures
import GetCBOWFeatures
import os


def focal_loss(gamma=2., alpha=.25):
    '''
    define the focal loss function
    :param gamma:
    :param alpha:
    :return: folcal loss
    '''
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def specific(confusion_matrix):
    '''recall = TP / (Tp + FN)'''
    specific = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
    return specific


def create_dl_model():
    '''
    create the deep-learning model
    :return: dl model
    '''
    model = keras.models.Sequential()
    model.add(Dense(units=505, input_dim=1011, kernel_initializer=glorot_uniform(111)))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dense(units=250, kernel_initializer=glorot_uniform(222)))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dense(units=125, kernel_initializer=glorot_uniform(222)))
    model.add(LeakyReLU())
    model.add(Dense(units=1, activation="sigmoid", kernel_initializer=glorot_uniform(333)))
    model.compile(loss=focal_loss(),
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def train_model(x, y, name):
    '''
    get the experimental results
    :param x: features
    :param y: labels
    :param name: Liao or Yu
    :return: results
    '''
    accs = []
    sps = []
    sns = []
    mccs = []
    pres = []
    aucs = []
    ys = []
    probs = []
    predictions = []
    if name == "Liao":
        n = 5
        print("the follow results is using the data segmentation strategy in Liao")
    elif name == "Yu":
        n = 10
    else:
        raise RuntimeError("Please check the value of parameter 'name'")
    for i in tqdm(range(20)):
        skf = StratifiedKFold(shuffle=True, n_splits=n)
        for train_index, test_index in skf.split(x, y):
            X_train, y_train = x[train_index], y[train_index]
            X_test, y_test = x[test_index], y[test_index]
            model = create_dl_model()
            model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=0)
            layermodel_1 = keras.models.Model(inputs=model.input, outputs=model.layers[2].output)
            X_train = layermodel_1.predict(X_train)
            X_test = layermodel_1.predict(X_test)
            gbdt = XGBClassifier(n_estimators=110, learning_rate=0.12)
            gbdt.fit(X_train, y_train.ravel())
            prediction = gbdt.predict(X_test)
            prob = gbdt.predict_proba(X_test)
            ys = ys + y_test.flatten().tolist()
            probs = probs + prob[:, 1].flatten().tolist()
            predictions = predictions + prediction.flatten().tolist()
            K.clear_session()
        cm = confusion_matrix(ys, predictions)
        acc = accuracy_score(ys, predictions)
        sp = specific(cm)
        sn = recall_score(ys, predictions, pos_label=1)
        mcc = matthews_corrcoef(ys, predictions)
        pre = precision_score(ys, predictions, pos_label=1)
        fpr_withoutAnn, tpr_withoutAnn, thresholds_withoutAnn = roc_curve(ys, probs)
        aucs.append(auc(fpr_withoutAnn, tpr_withoutAnn))
        pres.append(pre)
        accs.append(acc)
        sps.append(sp)
        sns.append(sn)
        mccs.append(mcc)
    if name == "Liao":
        print("acc: {:.4f}-{:.4f} ".format(np.mean(accs), np.std(accs)))
        print("s p: {:.4f}-{:.4f} ".format(np.mean(sps), np.std(sps)))
        print("s n: {:.4f}-{:.4f} ".format(np.mean(sns), np.std(sns)))
        print("mcc: {:.4f}-{:.4f} ".format(np.mean(mccs), np.std(mccs)))
        print("auc: {:.4f}-{:.4f} ".format(np.mean(aucs), np.std(aucs)))
    else:
        pass
    return np.mean(accs), np.mean(pres), np.mean(aucs)


def load_data():
    '''
    load datasets
    :return: datasets
    '''
    cbow_features = np.load("../files/CBOW_features.npy")
    bow_features = np.load("../files/BOW_features.npy")
    features = np.c_[cbow_features, bow_features]
    labels = pd.read_excel("../files/all_data.xlsx")["label"]
    return features, labels


if __name__ == "__main__":
    if os.path.exists('../files/CBOW_features'):
        pass
    else:
        GetCBOWFeatures.merge_features()
    if os.path.exists('../files/BOW_features'):
        pass
    else:
        GetBowFeatures.main()
    GetBowFeatures.main()
    x, y = load_data()
    train_model(x, y, name="Liao")

    print("the follow results is using the data segmentation strategy in Yu")
    positive_sample = x[np.where(y == 1)]
    neg_sample = x[np.where(y == 0)]
    np.random.shuffle(neg_sample)
    part_1 = neg_sample[0: 2597]
    part_2 = neg_sample[2597: 2597+2597]
    part_3 = neg_sample[2597+2597: 2597+2597+2596]
    part_4 = neg_sample[2597+2597+2596:]
    label_positive = np.ones((len(positive_sample), 1))
    x1 = np.r_[positive_sample, part_1]
    x2 = np.r_[positive_sample, part_2]
    x3 = np.r_[positive_sample, part_3]
    x4 = np.r_[positive_sample, part_4]
    y1 = np.r_[label_positive, np.zeros((len(part_1), 1))]
    y2 = np.r_[label_positive, np.zeros((len(part_2), 1))]
    y3 = np.r_[label_positive, np.zeros((len(part_3), 1))]
    y4 = np.r_[label_positive, np.zeros((len(part_4), 1))]
    acc = []
    pre = []
    aucs = []
    print("train model on the first sub-datasets.")
    acc_1, pre_1, auc_1 = train_model(x1, y1, name="Yu")
    acc.append(acc_1)
    pre.append(pre_1)
    aucs.append(auc_1)
    print("train model on the second sub-datasets.")
    acc_2, pre_2, auc_2 = train_model(x2, y2, name="Yu")
    acc.append(acc_2)
    pre.append(pre_2)
    aucs.append(auc_2)
    print("train model on the third sub-datasets.")
    acc_3, pre_3, auc_3 = train_model(x3, y3, name="Yu")
    acc.append(acc_3)
    pre.append(pre_3)
    aucs.append(auc_3)
    print("train model on the four sub-datasets.")
    acc_4, pre_4, auc_4 = train_model(x4, y4, name="Yu")
    acc.append(acc_4)
    pre.append(pre_4)
    aucs.append(auc_4)
    print("acc: ", np.mean(acc))
    print("pre: ", np.mean(pre))
    print("auc: ", np.mean(aucs))