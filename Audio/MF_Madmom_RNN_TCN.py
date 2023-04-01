import madmom
from madmom.features.beats import RNNBeatProcessor,  TCNBeatProcessor, DBNBeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from madmom.features.tempo import TCNTempoHistogramProcessor, DBNTempoHistogramProcessor, ACFTempoHistogramProcessor,TempoEstimationProcessor
from madmom.audio.signal import smooth

import mirdata
import librosa.display
import librosa
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from scipy.ndimage import maximum_filter1d
from scipy.interpolate import interp1d
from scipy.signal import argrelmax

class AudioFeatures(object):
    """
    Class containing methods to compute beat, DownBeat, tempo using madmom and librosa
    
    Attributes:
        hop_length: int
            Hop length between frames.  Same across all features
        sr: int
            Sample rate
        audio: ndarray()
            audio load
    """

    def __init__(self, audio,hop_length=512, sample_rate=44100):
        """
        
        Arguments:
            audio {[ndarray()]} -- [description]
            hop_length {int} -- [description] (default: {512})
            sample_rate {int} -- [description] (default: {44100})
        """
        self.hop_length = hop_length
        self.sr = sample_rate
        self.audio = audio
            

    def madmom_features(self, TCN = True, fps=100, beats_per_bar=[3, 4]):
        """
        Call Madmom's implementation of RNN/TCN + DBN (down) beat tracking. Madmom's
        results are returned in terms of seconds, but round and convert to
        be in terms of hop_size so that they line up with the features.

        Parameters
        ----------
        fps: int
            Frames per second in processing

        beats_per_bar: list
            list of beats per bar
        Returns
        -------
        output: a python dict with following key, value pairs
            {
                'beats': ndarray()
                    beat prediction
                'downbeats': ndarray()
                    downbeat prediction
            }
        """
        #Initialize DBN Beat Tracker to locate beats from beat activation
        beat_tracker = DBNBeatTrackingProcessor(fps=fps)                                     
        #Initialize DBN Downbeat Tracker to locate downbeats from downbeat activation
        downbeat_tracker = DBNDownBeatTrackingProcessor(beats_per_bar=beats_per_bar,fps=fps)
        if TCN == True:
          beat_activation = TCNBeatProcessor()(self.audio)
        else:
          beat_activation = RNNBeatProcessor()(self.audio)

        dbn_beat_pred = beat_tracker(beat_activation)
        # print("****BEAT PRED****")
        downbeat_activation = RNNDownBeatProcessor()(self.audio)[:,1]

        combined_act = np.vstack((np.maximum(beat_activation - downbeat_activation, 0), downbeat_activation)).T

        dbn_downbeat_pred = downbeat_tracker(combined_act)
        dbn_downbeat_pred = dbn_downbeat_pred[dbn_downbeat_pred[:, 1]==1][:, 0]
        # print("****DOWNBEAT PRED****")

        # bar
        beat_idx = (dbn_beat_pred * fps).astype(int)
        bar_act = maximum_filter1d(downbeat_activation, size=3)
        bar_act = bar_act[beat_idx]
        bar_act = np.vstack((dbn_beat_pred, bar_act)).T
        bar_tracker = madmom.features.downbeats.DBNBarTrackingProcessor(
                  beats_per_bar=beats_per_bar, meter_change_prob=1e-3, observation_weight=4)
        
        try:
            bars_pred = bar_tracker(bar_act)
        except IndexError:
            bars_pred = np.empty((0, 2))
        
        # print("****BARS PRED****")

        # if TCN == True:
        #     # Create a TCNTempoHistogramProcessor
        #     # Histogram (tuple of 2 numpy arrays, the first giving the strengths of
        #     # the bins and the second corresponding tempo/delay values)
        #     tempo_histogram = TCNTempoHistogramProcessor(fps=fps)(self.audio)

        #     # Get the most likely tempo
        #     # Numpy array with the dominant tempi [bpm] (first column) and their relative strengths (second column)
        #     tempo_pred = madmom.features.tempo.detect_tempo(tempo_histogram)
        # else:
        #     # tempo_histogram = ACFTempoHistogramProcessor(fps=fps)(self.audio)
        #     proc = TempoEstimationProcessor(fps=fps)
        #     tempo_pred = proc(beat_activation)

        # # print(tempo_pred, np.sum(tempo_pred,axis=0))

        # detections = {'beats': dbn_beat_pred, 'downbeats': dbn_downbeat_pred, 'bars': bars_pred, 'tempo':tempo_pred}
        tempo_pred = 1
        return dbn_beat_pred, dbn_downbeat_pred, tempo_pred

    def show_features(self,beat,downbeat,tempo, text = None, waveshow = False, spectrogram = False, bar=False, track= None, label = False,image_dir = 'image/'):
        """
        Show beat and downbeat detections with multiple way

        Parameters
        ----------
        waveshow: Bool
            plot on a waveshow

        Returns
        -------
            No returns
            except directly plot
        """

        FIGSIZE = (14,3)
        # print(f" audio duration : {self.audio.size*1/self.sr :.2f} seconds", self.sr)
        # print(beat, downbeat)
        if waveshow == True : 
            plt.figure(figsize=FIGSIZE)
            # read audio and annotations
            librosa.display.waveshow(self.audio, sr=self.sr, alpha=0.6)
            plt.vlines(beat, 1.1*self.audio.min(), 1.1*self.audio.max(), label='Beats', color='r', linestyle=':', linewidth=2)
            plt.vlines(downbeat, 1.1*self.audio.min(), 1.1*self.audio.max(), label=' Downbeats', color='r', linestyle='--', linewidth=2)
            if label == True:
                plt.vlines(track.beats.times, 1.1*self.audio.min(),1.1*self.audio.max(), label='Annotated beats', linestyle=':', color='g', linewidth=2)
                plt.vlines(track.beats.times[track.beats.positions == 1], 1.1*self.audio.min(), 1.1*self.audio.max(), label='Annotated downbeats',linestyle='--', color='g', linewidth=2)
            plt.legend(fontsize=12); 
            plt.title(f'Audio waveform with beat and downbeat predictions {text} BeatProcessor', fontsize=15)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.xlabel('Time (s)', fontsize=13)
            plt.legend(fontsize=12); 
            plt.xlim(20,30);
            plt.show()
            plt.savefig(f'{image_dir}/waveform_{text}.png')

        if spectrogram == True :
            plt.figure(figsize=FIGSIZE)
            S = librosa.feature.melspectrogram(self.audio, sr=self.sr, n_fft=2048,
                                   hop_length=self.hop_length,
                                   fmin=27.0,
                                   fmax=17000,
                                   n_mels=80)
            
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                         y_axis='mel', x_axis='time', sr=self.sr,
                         hop_length=self.hop_length, fmin=27.0, fmax=17000)

            plt.vlines(beat, 0, 17000, label='Beats', color='w', linestyle=':', linewidth=2)
            plt.vlines(downbeat, 0, 17000, label='Downbeats', color='w', linestyle='--', linewidth=2)
            if label == True:
                plt.vlines(track.beats.times, 0, 17000, label='Annotated beats', linestyles='dotted', linestyle=':', color='g', linewidth=2)
                plt.vlines(track.beats.times[track.beats.positions == 1], 0, 17000, label='Annotated downbeats', linestyle='--',color='g', linewidth=2)
            plt.title(f'Spectrogram with beat and downbeat predictions {text} BeatProcessor', fontsize=15)
            plt.xlim(20,30);
            plt.show()
            plt.savefig(f'{image_dir}/spectrogram_{text}.png')

        if  bar== True:
            plt.figure(figsize=FIGSIZE)
            tempo_bpm = tempo[:, 0]
            tempo_strength = tempo[:, 1]

            plt.bar(tempo_bpm, tempo_strength)
            plt.xlabel("Tempo (BPM)")
            plt.ylabel("Strength")
            plt.title("Tempo Histogram")
            plt.show()
            plt.savefig(f'{image_dir}/bar_plot_{text}.png')

if __name__ == "__main__":
    
    # print(madmom.__version__)
    gtzan = mirdata.initialize('gtzan_genre', version='mini')
    gtzan.download()
    tracks = gtzan.load_tracks()
    train_files, test_files = train_test_split(list(tracks.keys()), test_size=0.2, random_state=1234)
    song = test_files[-1] # 'pop.00002'

    track = tracks[song]
    AUDIO_DIR = track.audio_path

    # AUDIO_DIR =  "audio/Ed Sheeran - Shivers [Official Lyric Video].mp3"
    audio, sr = librosa.load(AUDIO_DIR, sr=44100)
    print(f" audio duration : {audio.size*1/sr :.2f} seconds", sr)
    feature = AudioFeatures(audio, sample_rate=sr)

    beat, downbeat, tempo = feature.madmom_features(fps = 100,TCN = False)

    beat_TCN, downbeat_TCN, tempo_TCN = feature.madmom_features(TCN = True)

    feature.show_features(beat, downbeat,tempo, text = 'RNN', waveshow = True, spectrogram = True)
    
    feature.show_features(beat_TCN, downbeat_TCN, tempo_TCN, text='TCN', waveshow = True, spectrogram = True)
