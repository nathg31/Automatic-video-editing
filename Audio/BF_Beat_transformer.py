import numpy as np 
import librosa
import torch
#Models to load
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
from BF_DilatedTransformer import Demixed_DilatedTransformerModel
import madmom
from madmom.features.beats import DBNBeatTrackingProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor

import mirdata

import matplotlib.pyplot as plt
import librosa.display

from sklearn.model_selection import train_test_split

class BeatTransformer(object):

    """
        Class containing methods to compute beat, DownBeat. Take from "BEAT TRANSFORMER: DEMIXED BEAT AND DOWNBEAT TRACKING
        WITH DILATED SELF-ATTENTION" Jingwei Zhao, Gus Xia, Ye Wang.
            
        Attributes:
            sr: int
                Sample rate

            audio: object
                path to the audio. No labels for the moment
            
            model : object
                model beat transformer
    """
    
    def __init__(self, audio_dir,sample_rate=44100):
        """
        Initialization of the attributes of our class.
        
        Arguments:
            audio {'object'} -- [path to the audio]
            sample_rate {int} -- [sample rate] (default: {44100})
        """
        self.sr = sample_rate
        self.audio_dir = audio_dir
    
    def demixed_audio(self,):
        """
        Creation of the spleeter instance, 
        we get 5 stems in output corresponding to C = {vocal; piano; drum; bass; other}. 
        For this we use Demixed_DilatedTransformerModel

        Parameters
        ----------

        -------
        output: 
            'x': ndarray()
                5 instruments channels, 5 spectrograms

        """
        #Initialize Spleeter for pre-processing (demixing)
        separator = Separator('spleeter:5stems')
        mel_f = librosa.filters.mel(sr=self.sr, n_fft=4096, n_mels=128, fmin=30, fmax=11000).T
        audio_loader = AudioAdapter.default()

        #Initialize Beat Transformer to estimate (down-)beat activation from demixed input

        waveform, _ = audio_loader.load(self.audio_dir, sample_rate=self.sr)

        x = separator.separate(waveform)
        x = np.stack([np.dot(np.abs(np.mean(separator._stft(x[key]), axis=-1))**2, mel_f) for key in x])
        x = np.transpose(x, (0, 2, 1))
        x = np.stack([librosa.power_to_db(x[i], ref=np.max) for i in range(len(x))])
        x = np.transpose(x, (0, 2, 1))
        return x
    
    def beat_transformer(self,x):
        """
        Creation of the spleeter instance, 
        we get 5 stems in output corresponding to C = {vocal; piano; drum; bass; other}. 
        For this we use Demixed_DilatedTransformerModel

        Parameters
        ----------

        -------
        output: 
            'dbn_beat_pred_beatnet': ndarray()
                array of beat prediction, value in seconds of beats

            'dbn_downbeat_pred_beatnet': ndarray()
                array of downbeat prediction, value in seconds of downbeat

        """
        model = Demixed_DilatedTransformerModel(attn_len=5, instr=5, ntoken=2, 
                                        dmodel=256, nhead=8, d_hid=1024, 
                                        nlayers=9,  norm_first=True)
        fold = 4
        param_path = {
            4: "./pretrained_models/fold_4_trf_param.pt"
        }
        model.load_state_dict(torch.load(param_path[fold], map_location=torch.device('cpu'))['state_dict'])
        if torch.cuda.is_available():
            model.cuda()
        model.eval()

        with torch.no_grad():
            if torch.cuda.is_available():
                model_input = torch.from_numpy(x).unsqueeze(0).float().cuda()
            else:
                model_input = torch.from_numpy(x).unsqueeze(0).float()

        activation, _ = model(model_input)

        dbn_beat_pred_post, dbn_downbeat_pred_post = self.post_processing(activation)

        return dbn_beat_pred_post, dbn_downbeat_pred_post

    def post_processing(self,activation, beat_per_bar = [3, 4]):
        """
        Creation of the spleeter instance, 
        we get 5 stems in output corresponding to C = {vocal; piano; drum; bass; other}. 
        For this we use Demixed_DilatedTransformerModel

        Parameters
        ----------
            'activation': ndarray()
                output of beat transformer

            beat_per_bar: list of int
                number of beats per bar 
        -------

        output: 
            'dbn_beat_pred_beatnet': ndarray()
                array of beat prediction, value in seconds of beats

            'dbn_downbeat_pred_beatnet': ndarray()
                array of downbeat prediction, value in seconds of downbeat

        """
        #Initialize DBN Beat Tracker to locate beats from beat activation
        beat_tracker = DBNBeatTrackingProcessor(min_bpm=55.0, max_bpm=215.0, fps=self.sr/1024, 
                                                transition_lambda=100, observation_lambda=6, 
                                                num_tempi=None, threshold=0.2)
                                                
        #Initialize DBN Downbeat Tracker to locate downbeats from downbeat activation
        downbeat_tracker = DBNDownBeatTrackingProcessor(beats_per_bar=beat_per_bar, 
                                                min_bpm=55.0, max_bpm=215.0, fps=self.sr/1024, 
                                                transition_lambda=100, observation_lambda=6, 
                                                num_tempi=None, threshold=0.2)

        beat_activation = torch.sigmoid(activation[0, :, 0]).detach().cpu().numpy()
        downbeat_activation = torch.sigmoid(activation[0, :, 1]).detach().cpu().numpy()
        dbn_beat_pred_post = beat_tracker(beat_activation)

        combined_act = np.concatenate((np.maximum(beat_activation - downbeat_activation, 
                                                np.zeros(beat_activation.shape)
                                                )[:, np.newaxis], 
                                    downbeat_activation[:, np.newaxis]
                                    ), axis=-1)   #(T, 2)

        dbn_downbeat_pred = downbeat_tracker(combined_act)
        dbn_downbeat_pred_post = dbn_downbeat_pred[dbn_downbeat_pred[:, 1]==1][:, 0]

        return dbn_beat_pred_post, dbn_downbeat_pred_post

    def show_features(self,beat,downbeat, text = None, waveshow = False, spectrogram = False, bar=False, track = None, label =False,image_dir = 'image/'):
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
        audio, sr = librosa.load(self.audio_dir)
        print(f" audio duration : {audio.size*1/sr :.2f} seconds", sr)
        print(beat, downbeat)
        if waveshow == True : 
            plt.figure(figsize=FIGSIZE)
            # read audio and annotations
            librosa.display.waveshow(audio, sr=sr, alpha=0.6)
            plt.vlines(beat, 1.1*audio.min(), 1.1*audio.max(), label='Beats', color='r', linestyle=':', linewidth=2)
            plt.vlines(downbeat, 1.1*audio.min(), 1.1*audio.max(), label=' Downbeats', color='r', linestyle='--', linewidth=2)
            if label == True:
                plt.vlines(track.beats.times, 1.1*audio.min(),1.1*audio.max(), label='Annotated beats', linestyle=':', color='g', linewidth=2)
                plt.vlines(track.beats.times[track.beats.positions == 1], 1.1*audio.min(), 1.1*audio.max(), label='Annotated downbeats',linestyle='--', color='g', linewidth=2)
            plt.legend(fontsize=12); 
            plt.title(f'Audio waveform with beat and downbeat predictions {text}', fontsize=15)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.xlabel('Time (s)', fontsize=13)
            plt.legend(fontsize=12); 
            plt.xlim(20, 30);
            plt.show()
            plt.savefig(f'{image_dir}/waveform_{text}.png')

        if spectrogram == True :
            plt.figure(figsize=FIGSIZE)
            S = librosa.feature.melspectrogram(audio, sr=sr, n_fft=2048,
                                   hop_length=512,
                                   fmin=27.0,
                                   fmax=17000,
                                   n_mels=80)
            
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                         y_axis='mel', x_axis='time', sr=sr,
                         hop_length=512, fmin=27.0, fmax=17000)

            plt.vlines(beat, 0, 17000, label='Beats', color='w', linestyle=':', linewidth=2)
            plt.vlines(downbeat, 0, 17000, label='Downbeats', color='w', linestyle='--', linewidth=2)
            if label == True:
                plt.vlines(track.beats.times, 0, 17000, label='Annotated beats', linestyles='dotted', linestyle=':', color='g', linewidth=2)
                plt.vlines(track.beats.times[track.beats.positions == 1], 0, 17000, label='Annotated downbeats', linestyle='--',color='g', linewidth=2)
            plt.title(f'Spectrogram with beat and downbeat predictions {text}', fontsize=15)
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
            plt.savefig(f'Image/bar_plot_{text}.png')

if __name__ == "__main__":

    gtzan = mirdata.initialize('gtzan_genre', version='mini')
    gtzan.download()
    tracks = gtzan.load_tracks()
    train_files, test_files = train_test_split(list(tracks.keys()), test_size=0.2, random_state=1234)
    song = test_files[-1] # 'pop.00002'

    track = tracks[song]
    AUDIO_DIR = track.audio_path

    model = BeatTransformer(AUDIO_DIR, 44100)
    x = model.demixed_audio()
    dbn_beat_pred, dbn_downbeat_pred = model.beat_transformer(x)

    print(dbn_beat_pred, dbn_downbeat_pred)

    model.show_features(dbn_beat_pred, dbn_downbeat_pred, text='transformer', waveshow = True, spectrogram = True)