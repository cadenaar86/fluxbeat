import librosa
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_audio(path: str, sr: int = 22050) -> np.ndarray:
    """
    Load an audio file as a floating point time series.
    
    Args:
        path (str): Path to audio file.
        sr (int): Target sampling rate.

    Returns:
        np.ndarray: Audio time series.
    """
    try:
        y, _ = librosa.load(path, sr=sr)
        return y
    except Exception as e:
        logger.error(f"Failed to load audio file {path}: {e}")
        return np.array([])
