import numpy as np
import librosa
from .. import config

class AudioProcessor:
    """
    Handles audio preprocessing: loading, segmentation, and normalization.
    """
    
    def __init__(self, sample_rate=config.SAMPLE_RATE, segment_duration=config.SEGMENT_DURATION):
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.samples_per_segment = int(sample_rate * segment_duration)

    def process_track(self, audio_array: np.ndarray):
        """
        Takes a full track audio array and returns a list of valid segments.
        
        Args:
            audio_array (np.ndarray): The full audio time series.
            
        Returns:
            np.ndarray: Array of shape (num_segments, samples_per_segment)
        """
        total_samples = len(audio_array)
        
        # If track is shorter than one segment, pad it
        if total_samples < self.samples_per_segment:
            padding = self.samples_per_segment - total_samples
            return np.array([np.pad(audio_array, (0, padding))])

        # Calculate number of segments
        num_segments = total_samples // self.samples_per_segment
        
        # Truncate to fit exact number of segments (ignoring the remainder for now)
        truncated_len = num_segments * self.samples_per_segment
        audio_truncated = audio_array[:truncated_len]
        
        # Reshape into segments
        segments = audio_truncated.reshape(num_segments, self.samples_per_segment)
        
        return segments

    def segment_sliding_window(self, audio_array: np.ndarray, overlap=0.5):
        """
        Segment audio with overlap (sliding window).
        Args:
            audio_array: Audio time series
            overlap: Float 0-1, fraction of overlap
        Returns:
            np.ndarray: (num_segments, samples_per_segment)
        """
        samples_per_segment = self.samples_per_segment
        stride = int(samples_per_segment * (1 - overlap))
        
        # Pad if shorter
        if len(audio_array) < samples_per_segment:
             padding = samples_per_segment - len(audio_array)
             return np.array([np.pad(audio_array, (0, padding))])
             
        # Extract windows
        # Using librosa util or manual striding
        # Manual striding for explicit control
        segments = []
        for start in range(0, len(audio_array) - samples_per_segment + 1, stride):
            end = start + samples_per_segment
            segments.append(audio_array[start:end])
            
        # Handle end of file (if last window didn't cover end)
        # Optional: Force capture last chunk with padding? 
        # For inference, users prefer covering everything.
        last_seg_end = segments[-1].shape[0] if len(segments) > 0 else 0
        # If we have significant leftover, pad and add
        # Simple approach: Standard sliding window usually ignores remainder if < window.
        
        if not segments:
             # Should be covered by pad above, but safe check
             return np.array([np.pad(audio_array, (0, samples_per_segment - len(audio_array)))])
             
        return np.array(segments)

    def normalize(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        Apply peak normalization.
        """
        max_val = np.abs(audio_segment).max()
        if max_val > 0:
            return audio_segment / max_val
        return audio_segment

    def compute_mel_spectrogram(self, audio_segment: np.ndarray):
        """
        Compute Mel Spectrogram for a given segment.
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio_segment,
            sr=self.sample_rate,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS
        )
        # Log-mel spectrogram
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec
