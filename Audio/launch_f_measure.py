import warnings
# Ignorer les avertissements au début du code
warnings.filterwarnings('ignore')

import mirdata
import librosa 
import os
from mir_eval.beat import f_measure

from segmenter import Segmenter
# import Madmom_features 
import Beat_transformer
# from DeepChorus import test

if __name__ == "__main__":

  ################# LOAD AUDIO #################
  print("***LOAD AUDIO***")
  gtzan = mirdata.initialize('gtzan_genre', version='mini')
  print("nombre de fichier audio dans gtzan mini :",len(gtzan.track_ids))
  tracks = gtzan.load_tracks()

  f_measure_beat = []
  f_measure_downbeat = []
  for name in tracks:
    print(name)
    track = tracks[name]
    AUDIO_DIR = track.audio_path
    # AUDIO_DIR = 'audio/Debussy.wav'
    # print("Path to the audio ",AUDIO_DIR)

    ################# SF SEGMENTER #################
    # print("***SF SEGMENTER***")
    # segmenter = Segmenter()
    # boundaries = segmenter.proc_audio(AUDIO_DIR, sr = 22050, is_label=False)
    # segmenter.plot(outdir='Image')
    # segmenter.refresh()
    # print('boundaries:', boundaries)

    ################# BEAT TRANSFORMER #################
    # print("***BEAT TRANSFORMER***")
    sr = 44100
    model = Beat_transformer.BeatTransformer(AUDIO_DIR, sr)
    x = model.demixed_audio()
    dbn_beat_pred, dbn_downbeat_pred = model.beat_transformer(x)

    reference_beats = track.beats.times
    f_measure_beat.append(f_measure(reference_beats, dbn_beat_pred))

    reference_downbeat = track.beats.times[track.beats.positions == 1]
    f_measure_downbeat.append(f_measure(reference_downbeat, dbn_downbeat_pred))

    # model.show_features(dbn_beat_pred, dbn_downbeat_pred, text='transformer', waveshow = True, spectrogram = True, track= track, label = True)


    ################# MADMOM #################
    # print("***MADMOM***")
    # sr = 44100
    # audio, _ = librosa.load(AUDIO_DIR, sr=sr)

    # feature = Madmom_features.AudioFeatures(audio, sample_rate=sr)

    # beat, downbeat, tempo = feature.madmom_features(fps = 100,TCN = False)

    # beat_TCN, downbeat_TCN, tempo_TCN = feature.madmom_features(TCN = True)

    # feature.show_features(beat, downbeat,tempo, text = 'RNN', waveshow = True, spectrogram = True, track= track, label = True)

    # feature.show_features(beat_TCN, downbeat_TCN, tempo_TCN, text='TCN', waveshow = True, spectrogram = True, track= track, label = True)


    ################# DeepChorus #################
    # print("***DEEPCHORUS***")
    # network_path = 'DeepChorus.network.DeepChorus'
    # model_path = 'DeepChorus/model/Deepchorus_2021.h5'
    # chorus_dict, chorus_dict_bin = test.deep_chorus(network_path, model_path, AUDIO_DIR)
    # print(chorus_dict)
    # print(chorus_dict_bin)
    # test.show_features(chorus_dict, chorus_dict_bin)

print(np.array(f_measure_beat).mean(), np.array(f_measure_downbeat).mean())