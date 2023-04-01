import warnings
# Ignorer les avertissements au début du code
warnings.filterwarnings('ignore')

import mirdata
import librosa 
import os
import numpy as np
from mir_eval.beat import trim_beats, f_measure
from SF_segmenter import Segmenter
import MF_Madmom_RNN_TCN
import MF_Madmom_RNN
import MF_Madmom_TCN
import BF_Beat_transformer
import DC_main


if __name__ == "__main__":

  ################# LOAD AUDIO #################
  print("***LOAD AUDIO***")
  gtzan = mirdata.initialize('gtzan_genre', version='mini')
  print("nombre de fichier audio dans gtzan mini :",len(gtzan.track_ids))
  tracks = gtzan.load_tracks()
  songs = ["pop.00000", "pop.00001", "pop.00002", "pop.00003",  "pop.00004", "pop.00005", "pop.00006", "pop.00007", "pop.00008", "pop.00009"]
  result_RNN = {}
  result_TCN = {}
  result_TRANSFO = {}
  cnt= 0
  # for track_id, _ in tracks.items():
  for track_id in songs :
    cnt+=1
    print(cnt)
  # for track_id in songs:
    track = tracks[track_id]
    AUDIO_DIR = track.audio_path
    # AUDIO_DIR = 'audio/Miss You-Oliver Tree & Robin Schulz.mp3'
    IMAGE_DIR = 'Image'

    ################# SF SEGMENTER #################
    # print("***SF SEGMENTER***")
    # segmenter = Segmenter()
    # boundaries = segmenter.proc_audio(AUDIO_DIR, sr = 22050, is_label=False)
    # segmenter.plot(outdir='Image')
    # segmenter.refresh()
    # print('boundaries:', boundaries)


    ################# BEAT TRANSFORMER #################
    print("***BEAT TRANSFORMER***")
    sr = 44100
    model = BF_Beat_transformer.BeatTransformer(AUDIO_DIR, sr)
    x = model.demixed_audio()
    dbn_beat_pred, dbn_downbeat_pred = model.beat_transformer(x)
    model.show_features(dbn_beat_pred, dbn_downbeat_pred, text='transformer', waveshow = True, spectrogram = True)


    ################# MADMOM #################
    # print("***MADMOM***")
    # sr = 44100
    # audio, _ = librosa.load(AUDIO_DIR, sr=sr)

    # feature  = MF_Madmom_RNN.AudioFeaturesRNN(audio, sample_rate =sr)
    # beat_RNN, downbeat_RNN, tempo_RNN = feature.madmom_features(fps = 100)
    # # feature.show_features(beat_RNN, downbeat_RNN,tempo_RNN, text = [track_id, 'RNN'], waveshow = True, spectrogram = False, label = True, track = track, image_dir=IMAGE_DIR)
    
    # feature  = MF_Madmom_TCN.AudioFeaturesTCN(audio, sample_rate =sr)
    # beat_TCN, downbeat_TCN, tempo_TCN= feature.madmom_features(fps = 100)
    # feature.show_features(beat_TCN, downbeat_TCN, tempo_TCN, text=[track_id, 'TCN'], waveshow = True, spectrogram = False, label = True, track = track, image_dir=IMAGE_DIR)
    # print(beat_RNN, track.beats.times)
    # print(beat_TCN)
    # print(downbeat_RNN, track.beats.times[track.beats.positions == 1])
    # print(downbeat_TCN)
    ################# DeepChorus #################
    # print("***DEEPCHORUS***")
    # network_path = 'pretrained_models.network.DeepChorus'
    # model_path = 'pretrained_models/Deepchorus_2021.h5'
    # chorus_dict, chorus_dict_bin = DC_main.deep_chorus(network_path, model_path, AUDIO_DIR)
    # print(chorus_dict)
    # print(chorus_dict_bin)
    # DC_main.show_features(chorus_dict, chorus_dict_bin,image_dir=IMAGE_DIR)

    # Calcul du décalage pour chaque modèle

    # offset_rnn= [np.abs(rnn - truth) for rnn, truth in zip(beat_RNN, track.beats.times)]
    # offset_tcn = [np.abs(tcn - truth) for tcn, truth in zip(beat_TCN, track.beats.times)]
    offset_transformer = [np.abs(trans - truth) for trans, truth in zip(dbn_beat_pred, track.beats.times)]

    # offset_rnn_down= [np.abs(rnn - truth) for rnn, truth in zip(downbeat_RNN, track.beats.times[track.beats.positions == 1])]
    # offset_tcn_down = [np.abs(tcn - truth) for tcn, truth in zip(downbeat_TCN, track.beats.times[track.beats.positions == 1])]
    offset_transformer_down = [np.abs(trans - truth) for trans, truth in zip(dbn_downbeat_pred, track.beats.times[track.beats.positions == 1])]


    # rnn_beat_list.append(np.mean(offset_rnn))
    # tcn_beat_list.append(np.mean(offset_tcn))
    # transfo_beat_list.append(np.mean(offset_transformer))
    # f_measure_beat_rnn = f_measure(trim_beats(track.beats.times), trim_beats(beat_RNN))
    # f_measure_beat_tcn = f_measure(trim_beats(track.beats.times), trim_beats(beat_TCN))
    f_measure_beat_transfo = f_measure(trim_beats(track.beats.times), trim_beats(dbn_beat_pred))

    # f_measure_downbeat_rnn = f_measure(trim_beats(track.beats.times[track.beats.positions == 1]), trim_beats(downbeat_RNN))
    # f_measure_downbeat_tcn = f_measure(trim_beats(track.beats.times[track.beats.positions == 1]), trim_beats(downbeat_TCN))
    f_measure_downbeat_transfo = f_measure(trim_beats(track.beats.times[track.beats.positions == 1]), trim_beats(dbn_downbeat_pred))

    # result_RNN[track_id] = {
    #     'offset_transformer': np.mean(offset_rnn),
    #     'f_measure_beat_transformer': f_measure_beat_rnn,
    #     'offset_down_transformer': np.mean(offset_rnn_down),
    #     'f_measure_downbeat_transformer': f_measure_downbeat_rnn
    # }
    # result_TCN[track_id] = {
    #     'offset_transformer': np.mean(offset_tcn),
    #     'f_measure_beat_transformer': f_measure_beat_tcn,
    #     'offset_down_transformer': np.mean(offset_tcn_down),
    #     'f_measure_downbeat_transformer': f_measure_downbeat_tcn
    # }
    result_TRANSFO[track_id] = {
        'offset_transformer': np.mean(offset_transformer),
        'f_measure_beat_transformer': f_measure_beat_transfo,
        'offset_down_transformer': np.mean(offset_transformer_down),
        'f_measure_downbeat_transformer': f_measure_downbeat_transfo
    }
    # print(" Décalage pour Beat RNN: ", np.mean(offset_rnn))
    # print(" Décalage pour Beat TCN: ", np.mean(offset_tcn))
    # print(" Décalage pour Beat Transformer: ", np.mean(offset_transformer))

    # print( " F measure Beat RNN BIS ", f_measure(trim_beats(track.beats.times), trim_beats(beat_RNN)))
    # print( " F measure Beat TCN ", f_measure(trim_beats(track.beats.times), trim_beats(beat_TCN)))
    # print( " F measure Beat Transformer ", f_measure(trim_beats(track.beats.times), trim_beats(dbn_beat_pred)))
  print(songs)
  # print("RNN = ", result_RNN)
  # print("TCN = ", result_TCN)
  print(" TRANSFO = ", result_TRANSFO)

  