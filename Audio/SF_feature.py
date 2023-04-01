import librosa
import numpy as np


def audio_extract_pcp(
        audio, 
        sr,
        n_fft=4096,
        hop_len=int(4096 * 0.75),
        pcp_bins=84,
        pcp_norm=np.inf,
        pcp_f_min=27.5,
        pcp_n_octaves=6):

    audio_harmonic, _ = librosa.effects.hpss(audio)
    pcp_cqt = np.abs(librosa.hybrid_cqt(
                audio_harmonic,
                sr=sr,
                hop_length=hop_len,
                n_bins=pcp_bins,
                norm=pcp_norm,
                fmin=pcp_f_min)) ** 2

    pcp = librosa.feature.chroma_cqt(
                C=pcp_cqt,
                sr=sr,
                hop_length=hop_len,
                n_octaves=pcp_n_octaves,
                fmin=pcp_f_min).T
    return pcp
