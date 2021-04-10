import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def encode_sequence(seq):
    '''
    encode gpcr sequence by AAindex
    :return: encoded sequence
    '''
    aaindex = pd.read_excel("../files/AAindex.xlsx")
    hydropathy_index = aaindex.iloc[0].values[1:22]
    hydropathy_index = hydropathy_index.astype(float)
    onehot_encoder = OneHotEncoder(sparse=False)
    meta_sequence = list(seq)
    meta_sequence_reshape = np.reshape(np.array(meta_sequence), (-1, 1))
    columns_name = list(aaindex.columns)
    columns_name.remove('         Amino Acid\nIndex Type')
    columns_name.remove('Attribute')
    columns_name.remove('Source')
    onehot_encoder.fit(np.reshape(columns_name, (-1, 1)))
    onehot_meta_sequence = onehot_encoder.transform(meta_sequence_reshape)
    meta_sequence_hydropathy = np.dot(hydropathy_index.reshape((1, -1)), onehot_meta_sequence.T)
    return meta_sequence_hydropathy.flatten()


def get_single_features(seq):
    '''
    extract bow features of a sequence
    :param seq: encoded sequence by AAindex
    :return: bow features
    '''
    chars = list(seq)
    list_aac = []
    acid = "ACDEFGHIKLMNPQRSTVWXY"
    acid = list(acid)
    counter = dict(Counter(chars).items())
    for i in acid:
        try:
            list_aac.append(counter[i] / len(chars))
        except:
            list_aac.append(0)
    aac_feature = np.reshape(np.array(list_aac), (1, -1))
    encode_seq = encode_sequence(seq)
    bow_b = np.load('../files/bowOfB.npy')
    bow_c = np.load('../files/bowOfC.npy')
    bow_d = np.load('../files/bowOfD.npy')
    meta_array_b = np.array([])
    for i in range(encode_seq.shape[0]):
        if (i != encode_seq.shape[0] - 1):
            meta_array_b = np.r_[meta_array_b, np.array(encode_seq[i: i + 2])]
    seq_length_2 = meta_array_b.reshape((-1, 2))

    temp_b = 0
    for i in range(16):
        if i == 0:
            temp_b = np.sum(np.power(seq_length_2 - bow_b[i], 2), axis=1)
        else:
            temp_b = np.c_[temp_b, np.sum(np.power(seq_length_2 - bow_b[i], 2), axis=1)]
    list_b = []
    b = dict(Counter(np.argmin(temp_b, axis=1)))
    for i in range(16):
        if i not in sorted(b):
            list_b.append(0)
        else:
            list_b.append((b[i] / seq_length_2.shape[0]))
    wb_b_feature = np.reshape(np.array(list_b), (1, -1))

    meta_array_c = np.array([])
    for i in range(encode_seq.shape[0]):
        if (i < encode_seq.shape[0] - 2):
            meta_array_c = np.r_[meta_array_c, np.array(encode_seq[i: i + 3])]
    seq_length_3 = meta_array_c.reshape((-1, 3))
    temp_c = 0
    for i in range(62):
        if i == 0:
            temp_c = np.sum(np.power(seq_length_3 - bow_c[i], 2), axis=1)
        else:
            temp_c = np.c_[temp_c, np.sum(np.power(seq_length_3 - bow_c[i], 2), axis=1)]
    list_c = []
    c = dict(Counter(np.argmin(temp_c, axis=1)))
    for i in range(62):
        if i not in sorted(c):
            list_c.append(0)
        else:
            list_c.append((c[i] / seq_length_3.shape[0]))
    wb_c_feature = np.reshape(np.array(list_c), (1, -1))

    meta_array_d = np.array([])
    for i in range(encode_seq.shape[0]):
        if (i < encode_seq.shape[0] - 2):
            meta_array_d = np.r_[meta_array_d, np.array(encode_seq[[i, i + 2]])]
    seq_length_interval = meta_array_d.reshape((-1, 2))
    temp_d = 0
    for i in range(16):
        if i == 0:
            temp_d = np.sum(np.power(seq_length_interval - bow_d[i], 2), axis=1)
        else:
            temp_d = np.c_[temp_d, np.sum(np.power(seq_length_interval - bow_d[i], 2), axis=1)]
    list_d = []
    d = dict(Counter(np.argmin(temp_d, axis=1)))
    for i in range(16):
        if i not in sorted(d):
            list_d.append(0)
        else:
            list_d.append((d[i] / seq_length_interval.shape[0]))
    wb_d_feature = np.reshape(np.array(list_d), (1, -1))
    gpcr_feature = np.concatenate((aac_feature, wb_b_feature, wb_c_feature, wb_d_feature), axis=1)
    return gpcr_feature


def get_bow(data):
    '''
    create the bow models
    :param data: datasets
    '''
    data["sequence_b"] = None
    data["sequence_c"] = None
    data["sequence_d"] = None
    sequence_b_total = np.array([])
    sequence_c_total = np.array([])
    sequence_d_total = np.array([])
    print("Get bow ...")
    for index in tqdm(range(len(data))):
        meta_array_B = np.array([])
        meta_array_C = np.array([])
        meta_array_D = np.array([])
        value = encode_sequence(data.loc[index, "seq"])
        for i in range(value.shape[0]):
            if (i < value.shape[0] - 1):
                meta_array_B = np.r_[meta_array_B, np.array(value[i: i + 2])]
        meta_array_B = meta_array_B.reshape((-1, 2))
        if index == 0:
            sequence_b_total = meta_array_B
        else:
            sequence_b_total = np.r_[sequence_b_total, meta_array_B]
        for i in range(value.shape[0]):
            if (i < value.shape[0] - 2):
                meta_array_C = np.r_[meta_array_C, np.array(value[i: i + 3])]
        meta_array_C = meta_array_C.reshape((-1, 3))
        if index == 0:
            sequence_c_total = meta_array_C
        else:
            sequence_c_total = np.r_[sequence_c_total, meta_array_C]
        for i in range(value.shape[0]):
            if (i < value.shape[0] - 2):
                meta_array_D = np.r_[meta_array_D, np.array(value[[i, i + 2]])]
        meta_array_D = meta_array_D.reshape((-1, 2))
        if index == 0:
            sequence_d_total = meta_array_D
        else:
            sequence_d_total = np.r_[sequence_d_total, meta_array_D]
    bow_b = MiniBatchKMeans(n_clusters=16)
    bow_c = MiniBatchKMeans(n_clusters=62)
    bow_d = MiniBatchKMeans(n_clusters=16)
    bow_b.fit(sequence_b_total)
    bow_c.fit(sequence_c_total)
    bow_d.fit(sequence_d_total)
    np.save("../files/bowOfB.npy", bow_b.cluster_centers_)
    np.save("../files/bowOfC.npy", bow_c.cluster_centers_)
    np.save("../files/bowOfD.npy", bow_d.cluster_centers_)


def get_all_fetures(data):
    '''
    extract bow features
    :param data: datasets
    :return: bow features
    '''
    all_features = None
    length = len(data)
    print("Get bow features...")
    for index in tqdm(range(length)):
        sequence = data.loc[index, "seq"]
        if index == 0:
            all_features = get_single_features(sequence)
        else:
            all_features = np.r_[all_features, get_single_features(sequence)]
    np.save("../files/BOW_features.npy", all_features)
    return all_features


def replace(string: str):
    string = string.replace('B', '')
    string = string.replace('U', '')
    string = string.replace('Z', '')
    string = string.replace('O', '')
    return string


def main():
    path = "../files/all_data.xlsx"
    data = pd.read_excel(path)
    data.seq = data['seq'].apply(replace)
    get_bow(data)
    features = get_all_fetures(data)
    return features


if __name__ == '__main__':
    features = main()