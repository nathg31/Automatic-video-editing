import copy
from itertools import count
import numpy as np
import os
from tensorflow.keras import Input, Model
from DC_constant import *
import tensorflow as tf



def scale_result(result):
    new_np = np.zeros(len(result))
    max_score = np.max(result)
    min_score = np.min(result)
    overlap = max_score - min_score
    for i in range(len(result)):
        new_np[i] = (result[i] - min_score) / overlap
    return new_np


def median_filter(result_np, sample):

    begin = side = int(sample // 2)
    for i in range(begin):
        result_np = np.insert(result_np, 0, result_np[2 * i])
        result_np = np.append(result_np, result_np[-(2 * i)])

    filtered_np = copy.deepcopy(result_np)

    for i in range(begin, (len(filtered_np) - side)):
        group_s = i - side  # +2
        group_e = i + side +1 # -1
        window = result_np[group_s: group_e]
        print('window', window)
        r_max = np.max(window)
        r_min = np.min(window)
        mid_value = float((sum(window) - r_min - r_max) / (len(window) - 2))
        filtered_np[i] = mid_value
    for i in range(begin):
        filtered_np = np.delete(filtered_np, 0)
        filtered_np = np.delete(filtered_np, len(filtered_np) - 1)
    return filtered_np


def get_result_dict(model, features_dict):

    @tf.function(experimental_relax_shapes=True)
    def predict(t):
        return model(t)
    raw_dict = {}
    bin_dict = {}
    for key in features_dict.keys():
        
        feature = features_dict[key]
        feature_length = len(feature[1])
        remain = feature_length % 9
        remain_np = np.zeros([128, 9 - remain, 1])
        feature_crop = np.concatenate((feature, remain_np), axis=1)
        feature_crop = np.expand_dims(feature_crop, axis=0)

        result_np = predict(feature_crop)[0]
        result_np = scale_result(result_np)
        result_np = median_filter(result_np, 9)
        raw_dict[key] = copy.deepcopy(result_np)
        for i in range(len(result_np)):
            if result_np[i] < 0.5: # TRESHOLD
                result_np[i] = 0
            else:
                result_np[i] = 1
        bin_dict[key] = copy.deepcopy(result_np)

    return raw_dict, bin_dict
