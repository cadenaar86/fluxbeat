import librosa
import numpy as np
import torch
from .. import config

class FeatureExtractor:
    """
    Handles extraction of audio features for the FluxBeat model.
    """
    def __init__(self, sample_rate=config.SAMPLE_RATE, n_fft=config.N_FFT, 
                 hop_length=config.HOP_LENGTH, n_mels=config.N_MELS):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def compute_mel_spectrogram(self, audio: np.ndarray, to_db=True):
        """
        Compute Mel Spectrogram.
        Returns shape: (n_mels, time_steps)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        if to_db:
            return librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec

    def compute_mfcc(self, audio: np.ndarray, n_mfcc=13):
        """
        Compute MFCCs.
        Returns shape: (n_mfcc, time_steps)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return mfcc

    def compute_chroma(self, audio: np.ndarray):
        """
        Compute Chroma STFT.
        Returns shape: (12, time_steps)
        """
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return chroma

    def compute_spectral_contrast(self, audio: np.ndarray):
        """
        Compute Spectral Contrast.
        Returns shape: (7, time_steps) usually 
        (number of bands + 1)
        """
        contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return contrast

    def compute_all(self, audio: np.ndarray):
        """
        Compute a dictionary of all features.
        """
        return {
            'mel_spectrogram': self.compute_mel_spectrogram(audio),
            'mfcc': self.compute_mfcc(audio),
            'chroma': self.compute_chroma(audio),
            'spectral_contrast': self.compute_spectral_contrast(audio)
        }
