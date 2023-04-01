import madmom
from madmom.features.beats import   TCNBeatProcessor, DBNBeatTrackingProcessor
from madmom.features.downbeats import  RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from madmom.features.tempo import TCNTempoHistogramProcessor, TempoEstimationProcessor
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
from scipy.signal import find_peaks

class AudioFeaturesTCN(object):
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
            

    def madmom_features(self, fps=100, beats_per_bar=[3, 4]):
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

        beat_activation = TCNBeatProcessor()(self.audio)


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

        # Create a TCNTempoHistogramProcessor
        # Histogram (tuple of 2 numpy arrays, the first giving the strengths of
        # the bins and the second corresponding tempo/delay values)
        # tempo_histogram = TCNTempoHistogramProcessor(fps=fps)(self.audio)

        # Get the most likely tempo
        # Numpy array with the dominant tempi [bpm] (first column) and their relative strengths (second column)
        # tempo_pred = madmom.features.tempo.detect_tempo(tempo_histogram)

        tempo_pred = self.estimate_tempo(beat_activation,min_bpm=70, max_bpm=200)
        # print(tempo_pred, np.sum(tempo_pred,axis=0))

        detections = {'beats': dbn_beat_pred, 'downbeats': dbn_downbeat_pred, 'bars': bars_pred, 'tempo':tempo_pred}
        return dbn_beat_pred, dbn_downbeat_pred, tempo_pred

    def estimate_tempo(self,beat, fps=100,min_bpm=None, max_bpm=None):
        """
        Estimate tempo using Madmom's TCNTempoHistogramProcessor.

        Parameters
        ----------
        fps: int
            Frames per second in processing
        beat: array
            Array of beat detection, value in seconds
        Returns
        -------
        tempo_estimation: float
            Estimated tempo in BPM
        """

        # Estimate tempo
        tempo_hist_proc = TCNTempoHistogramProcessor(min_bpm=min_bpm, max_bpm=max_bpm)
        tempo_histogram, tempi = tempo_hist_proc(beat)

        # Find peaks in the tempo_histogram with a minimum prominence
        peaks, properties = find_peaks(tempo_histogram, prominence=0.01)
        # Find peaks in the tempo_histogram with a minimum prominence
        peaks, properties = find_peaks(tempo_histogram, prominence=0.01)

        # Create a list of tuples containing the peak and its prominence
        peak_prominence = list(zip(peaks, properties['prominences']))

        # Sort the list of tuples by prominence (in descending order)
        sorted_peaks = sorted(peak_prominence, key=lambda x: x[1], reverse=True)

        # Extract the sorted peak values
        sorted_peaks_values = [peak for peak, prominence in sorted_peaks]

        # Return the top 'num_peaks' peaks and their corresponding tempi
        significant_peaks = sorted_peaks_values[:5]
        significant_tempi = [tempi[p] for p in significant_peaks]
        self.tempo_histogram = tempo_histogram
        return significant_tempi

    def show_features(self,beat,downbeat, text = None, waveshow = False, spectrogram = False, tempo=None, track= None, label = False,image_dir = 'image/'):
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
            plt.xlim(30,60);
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
            plt.xlim(30,60);
            plt.show()
            plt.savefig(f'{image_dir}/spectrogram_{text}.png')

        if tempo == True:
            tempi = np.arange(70, 201)
            plt.figure(figsize=(10, 5))
            plt.bar(tempi, self.tempo_histogram, width=1)
            plt.xlabel('Tempo (BPM)')
            plt.ylabel('Strength')
            plt.title('Tempo Histogram')
            plt.grid(True)
            plt.savefig(f'{image_dir}/tempo_histo_{text}.png')
            plt.show()

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
    feature = AudioFeaturesTCN(audio, sample_rate=sr)


    beat_TCN, downbeat_TCN, tempo_TCN = feature.madmom_features()

    
    feature.show_features(beat_TCN, downbeat_TCN, tempo_TCN, text='TCN', waveshow = True, spectrogram = True)
