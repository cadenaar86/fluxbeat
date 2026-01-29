import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import os
import cv2
from .. import config

def plot_spectrogram(spectrogram, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH, 
                    title="Spectrogram", save_path=None):
    """
    Plot and optionally save a spectrogram.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_wave(audio, sr=config.SAMPLE_RATE, title="Waveform", save_path=None):
    """
    Plot waveform.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_chroma(chroma, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH, 
               title="Chroma", save_path=None):
    """
    Plot chroma features.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', 
                             sr=sr, hop_length=hop_length)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
