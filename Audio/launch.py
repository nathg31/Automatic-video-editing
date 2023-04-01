import warnings
# Ignorer les avertissements au début du code
warnings.filterwarnings('ignore')

import mirdata
import librosa 
import os

from SF_segmenter import Segmenter
# import MF_Madmom_RNN
import MF_Madmom_TCN
# import BF_Beat_transformer
# import DC_main

import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
  result_TRANSFO = {}
  ################# LOAD AUDIO #################
  print("***LOAD AUDIO***")
  gtzan = mirdata.initialize('gtzan_genre', version='mini')
  # print("nombre de fichier audio dans gtzan mini :",len(gtzan.track_ids))
  tracks = gtzan.load_tracks()
  song = "pop.00002"
  track = tracks[song]
  # print(track)
  AUDIO_DIR = track.audio_path
  AUDIO_DIR = 'audio/Flowers-Miley Cyrus.mp3'
  IMAGE_DIR = 'Image/'

  ################# SF SEGMENTER #################
  # print("***SF SEGMENTER***")
  # segmenter = Segmenter()
  # boundaries = segmenter.proc_audio(AUDIO_DIR, sr = 22050, is_label=False)
  # segmenter.plot(outdir='Image')
  # segmenter.refresh()
  # print('boundaries:', boundaries)

  ################# BEAT TRANSFORMER #################
  # print("***BEAT TRANSFORMER***")
  # sr = 44100
  # model = BF_Beat_transformer.BeatTransformer(AUDIO_DIR, sr)
  # x = model.demixed_audio()
  # dbn_beat_pred, dbn_downbeat_pred = model.beat_transformer(x)
  # print(dbn_beat_pred, dbn_downbeat_pred)
  # model.show_features(dbn_beat_pred, dbn_downbeat_pred, text='transformer', waveshow = True, spectrogram = True,image_dir=IMAGE_DIR)


  ################# MADMOM #################
  print("***MADMOM***")
  sr = 44100
  audio, _ = librosa.load(AUDIO_DIR, sr=sr)

  # # feature  = MF_Madmom_RNN.AudioFeaturesRNN(audio, sample_rate =sr)
  # # beat_RNN, downbeat_RNN, tempo_RNN = feature.madmom_features(fps = 100)
  # # feature.show_features(beat_RNN, downbeat_RNN,tempo_RNN, text = 'RNN', waveshow = True, spectrogram = True,  image_dir=IMAGE_DIR)
  
  feature  = MF_Madmom_TCN.AudioFeaturesTCN(audio, sample_rate =sr)
  beat_TCN, downbeat_TCN, tempo_TCN= feature.madmom_features(fps = 100)
  print('tempo_TCN', tempo_TCN)
  feature.show_features(beat_TCN, downbeat_TCN, text = 'TCN', waveshow = True, spectrogram = True, tempo=True,image_dir = IMAGE_DIR)
  # print(beat_RNN, track.beats.times)
  # print(beat_TCN)
  # print(downbeat_RNN)
  # print(downbeat_TCN)


  ################# DeepChorus #################
  # print("***DEEPCHORUS***")
  # network_path = 'pretrained_models.network.DeepChorus'
  # model_path = 'pretrained_models/Deepchorus_2021.h5'
  # chorus_dict, chorus_dict_bin = DC_main.deep_chorus(network_path, model_path, AUDIO_DIR)
  # DC_main.show_features(chorus_dict, chorus_dict_bin, image_dir = IMAGE_DIR)

  # print(chorus_dict)
  # print(chorus_dict_bin) 
  # dbn_beat_pred = np.load('dbn_beat_pred_beatnet.npy')
  # np.save('beat_TCN', beat_TCN)
  # np.save('downbeat_TCN', downbeat_TCN)
  # np.save('tempo_TCN', tempo_TCN)
  # np.save('downbeat_RNN',downbeat_RNN)
  # np.save('downbeat_TCN',downbeat_TCN)
  # downbeat_TCN = np.load('downbeat_TCN.npy')
  # beat_TCN = np.load('beat_TCN.npy')
  # tempo_TCN = np.load('tempo_TCN.npy')
  # downbeat_TCN = np.load('downbeat_TCN.npy')

  # dbn_downbeat_pred = np.load('dbn_downbeat_pred_beatnet.npy')
  # print("RNN", downbeat_RNN)
  # print("TCN", downbeat_TCN)
  # print("Transfo", dbn_downbeat_pred)

  # print("RNN", downbeat_RNN.shape)
  # print("TCN", downbeat_TCN.shape)
  # print("Transfo", dbn_downbeat_pred.shape)

  # def find_closest(array, value):
  #   idx = np.argmin(np.abs(array - value))
  #   return array[idx], idx

  # diff1 = []
  # diff2 = []
  # for rnn_time in downbeat_RNN:
  #     closest_tcn_time, tcn_idx = find_closest(downbeat_TCN, rnn_time)
  #     closest_transfo_time, transfo_idx = find_closest(dbn_downbeat_pred, rnn_time)
  #     diff1.append(closest_transfo_time - rnn_time)
  #     diff2.append(closest_tcn_time - rnn_time)


  # # Calcul des différences temporelles
  # diff1 = dbn_downbeat_pred - downbeat_RNN[:-2]
  # diff2 = downbeat_TCN[:-2] - downbeat_RNN[:-2]

  # Création du graphique
  # fig, ax = plt.subplots()
  # ax.plot(diff1, label='Transformer', marker='o')
  # ax.plot(diff2, label='TCN', marker='o')

  # # Configuration du graphique
  # ax.set_xlabel('Index')
  # ax.set_ylabel('Différence temporelle')
  # ax.set_title('Différences temporelles Downbeat référence (RNN)')
  # ax.legend()

  # Sauvegarde du graphique
  # plt.savefig('differences_temporelles_downbeat_missyou.png')


  # dbn_beat_pred = np.load('dbn_beat_pred_beatnet.npy')

  # print("RNN", beat_RNN.shape)
  # print("TCN", beat_TCN.shape)
  # print("Transfo", dbn_beat_pred.shape)

  # diff1 = []
  # diff2 = []
  # for rnn_time in beat_RNN:
  #     closest_tcn_time, tcn_idx = find_closest(beat_TCN, rnn_time)
  #     closest_transfo_time, transfo_idx = find_closest(dbn_beat_pred, rnn_time)
  #     diff1.append(closest_transfo_time - rnn_time)
  #     diff2.append(closest_tcn_time - rnn_time)

  # # Calcul des différences temporelles
  # diff1 = dbn_beat_pred - beat_RNN[:-7]
  # diff2 = beat_TCN[:-8] - beat_RNN[:-7]

  # Création du graphique
  # fig, ax = plt.subplots()
  # ax.plot(diff1, label='Transformer', marker='o')
  # ax.plot(diff2, label='TCN', marker='o')

  # # Configuration du graphique
  # ax.set_xlabel('Index')
  # ax.set_ylabel('Différence temporelle')
  # ax.set_title('Différences temporelles Beat référence (RNN)')
  # ax.legend()

  # # Sauvegarde du graphique
  # plt.savefig('differences_temporelles_beat_missyou.png')
  # DC_main.show_features(chorus_dict, chorus_dict_bin,image_dir=IMAGE_DIR)
  # result_TRANSFO['Ed Sheeran - Shivers - 90s'] = {
  # 'Beat': dbn_beat_pred,
  # 'Downbeat': dbn_downbeat_pred,
  # 'Chorus': np.array(list(chorus_dict.values())[0])
  # }
  
  # # Save the dictionary to a file
  # with open('metadata_shivers_90s.pickle', 'wb') as handle:
  #     pickle.dump(result_TRANSFO, handle, protocol=pickle.HIGHEST_PROTOCOL)
  # print(result_TRANSFO)
