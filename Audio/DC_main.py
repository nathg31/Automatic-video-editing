import os
import argparse
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import joblib
import importlib
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from DC_constant import *
from DC_evaluator import get_result_dict


def deep_chorus(network_path, model_path, audio_file):
    # Check if GPU is available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Training on GPU...")
    else:
        print("Training on CPU...")


    y, _ = librosa.load(audio_file, sr=SR)
    # Find the length of the audio in seconds
    audio_length = len(y) / SR

    print(f"The length of the audio file is {audio_length:.2f} seconds.")
    
    data = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=N_HOP,
                                            power=1, n_mels=N_MEL, fmin=20, fmax=5000)
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    key = os.path.splitext(os.path.basename(audio_file))[0]

    # Loading Network
    current_dir = os.getcwd()
    print("Current working directory:", current_dir)
    network_module = importlib.import_module(network_path)
    create_model = network_module.create_model

    # Loading Model
    model = create_model(input_shape=SHAPE, chunk_size=CHUNK_SIZE)
    model.compile(loss='binary_crossentropy', optimizer=(Adam(learning_rate=LR)),
                metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5), tf.keras.metrics.Recall()])
    model.load_weights(model_path)

    # Loading Data
    features = {key: data}

    # Predict Result
    predictions_dict,predictions_dict_bin = get_result_dict(model, features)    
    return predictions_dict, predictions_dict_bin

def show_features(dict1, dict2, image_dir='image/'):
    # Loop over the dictionaries and plot the values
    for key, values in dict1.items():
        fig, ax = plt.subplots()
        ax.plot(values, label=key)
        ax.axhline(y=0.5, color='r', linestyle='--')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Chorus Probability')
        ax.set_title(f'Chorus Detection')
        ax.legend()
        plt.savefig(f'{image_dir}/DeepChorus_{key}.png')
        plt.show()

    for key, values in dict2.items():
        fig, ax = plt.subplots()
        ax.plot(values, label=key)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Value binarized')
        ax.set_title(f'Chorus Detection binarized')
        ax.legend()
        plt.savefig(f'{image_dir}/DeepChorus_binarized_{key}.png')
        plt.show()