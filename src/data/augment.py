import torch
import torch.nn as nn
import numpy as np
import torchaudio.transforms as T

class AudioAugmentor:
    """
    Augmentation pipeline for Self-Supervised Learning.
    Applies transforms to Spectrograms or Raw Audio.
    """
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
        # Raw Audio Transforms
        # Note: Time stretch / Pitch shift in time domain are slow.
        # We focus on Spectrogram Augmentation (SpecAugment) which is standard.
        
        # SpecAugment
        self.time_masking = T.TimeMasking(time_mask_param=10)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=10)
        
    def add_noise(self, audio, noise_level=0.005):
        noise = torch.randn_like(audio) * noise_level
        return audio + noise

    def apply_spectrogram_augment(self, spec):
        """
        Apply SpecAugment (Time/Freq Masking).
        spec: (Batch, Channel, Freq, Time) or (Channel, Freq, Time)
        """
        # Torchaudio transforms expect (..., Freq, Time)
        out = self.freq_masking(spec)
        out = self.time_masking(out)
        return out

    def __call__(self, spec):
        return self.apply_spectrogram_augment(spec)

class SimCLRAugment:
    """
    Returns two augmented views of the same input.
    """
    def __init__(self, augmentor):
        self.augmentor = augmentor
        
    def __call__(self, x):
        return self.augmentor(x), self.augmentor(x)
