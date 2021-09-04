import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import glorot_uniform
from xgboost import XGBClassifier
import numpy as np
import GetBowFeatures
import GetCBOWFeatures


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


def train_model():
    x, y = load_data()
    model = create_dl_model()
    model.fit(x, y, epochs=20, batch_size=32, verbose=0, callbacks=keras.callbacks.EarlyStopping(monitor='loss', patience=3))
    layermodel_1 = keras.models.Model(inputs=model.input, outputs=model.layers[2].output)
    x = layermodel_1.predict(x)
    xgb = XGBClassifier(n_estimators=110, learning_rate=0.12)
    xgb.fit(x, y.ravel())
    layermodel_1.save("../files/DNN.h5")
    xgb.save_model("../files/xgboost.model")


def load_data():
    '''
    load datasets
    :return: datasets
    '''
    cbow_features = GetCBOWFeatures.merge_features()
    bow_features = GetBowFeatures.main()
    features = np.c_[cbow_features, bow_features]
    labels = pd.read_excel("../files/all_data.xlsx")["label"]
    return features, labels


if __name__ == "__main__":
    train_model()