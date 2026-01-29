import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from .. import config
from .processor import AudioProcessor
from ..utils.audio import load_audio

class FMADataset(Dataset):
    """
    Dataset class for Free Music Archive (FMA).
    """
    def __init__(self, metadata_dir=config.METADATA_DIR, subset='small', split='train', 
                 transform=None, mode='segment', cache=False):
        """
        Args:
            metadata_dir (str): Path to metadata directory.
            subset (str): 'small', 'medium', 'large', 'full'.
            split (str): 'train', 'validation', 'test'.
            transform (callable): Optional transform to be applied.
            mode (str): 'full' (return full track) or 'segment' (return segments).
            cache (bool): If True, cache computed features to disk.
        """
        self.metadata_dir = metadata_dir
        self.subset = subset
        self.split = split
        self.transform = transform
        self.mode = mode
        self.cache = cache
        
        self.processor = AudioProcessor()
        # Initializing extractor internally if caching is on or needed
        from .features import FeatureExtractor
        self.feature_extractor = FeatureExtractor()
        
        # Load tracks metadata
        self.tracks = self._load_metadata()
        self.config_paths()
        
        if self.cache:
            self.cache_dir = os.path.join(config.CACHE_DIR, self.subset)
            os.makedirs(self.cache_dir, exist_ok=True)

    def _load_metadata(self):
        tracks_path = os.path.join(self.metadata_dir, 'tracks.csv')
        if not os.path.exists(tracks_path):
            raise FileNotFoundError(f"Metadata not found at {tracks_path}. Please run downloader.")
            
        # FMA tracks.csv has a weird header (3 lines)
        tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1, 2])
        
        # Filter by subset
        if self.subset != 'full':
            subset_mask = tracks[('set', 'subset', '')] == self.subset
            tracks = tracks[subset_mask]
            
        # Filter by split
        split_mask = tracks[('set', 'split', '')] == self.split
        tracks = tracks[split_mask]
        
        return tracks

    def config_paths(self):
        # We need to reconstruct file paths.
        # FMA audio files are stored in: {subset}/{000}/{000000}.mp3 usually
        # But locally we might organize them as data/raw/fma_{subset}/{000}/{track_id}.mp3
        pass

    def __len__(self):
        return len(self.tracks)

    def get_audio_path(self, track_id):
        # Helper to find audio file
        # ID is int, formatted as 6 digit string
        tid_str = '{:06d}'.format(track_id)
        # raw/fma_{subset}/{tid_str[:3]}/{tid_str}.mp3 
        # Note: This directory structure depends on how we extract the zip. 
        # FMA zip extracts to fma_{subset}/{tid_str[:3]}/{tid_str}.mp3
        return os.path.join(config.RAW_DATA_DIR, f"fma_{self.subset}", tid_str[:3], f"{tid_str}.mp3")

    def get_cache_path(self, track_id):
        return os.path.join(self.cache_dir, f"{track_id}.npy")

    def __getitem__(self, idx):
        track_id = self.tracks.index[idx]
        
        # Labels - for 'small' dataset, use 'track.genre_top'
        try:
            label = self.tracks.loc[track_id, ('track', 'genre_top', '')]
        except:
            label = "Unknown"
            
        # Try Cache Hit
        features = None
        if self.cache:
            cache_path = self.get_cache_path(track_id)
            if os.path.exists(cache_path):
                try:
                    features = np.load(cache_path)
                    # Features shape: (Num_Segments, Freq, Time) (Mel Spec) 
                    # Note: We saved without Channel dim usually, so (N, F, T). 
                    # Model expects (B, C, F, T).
                except Exception as e:
                    print(f"Error loading cache for {track_id}: {e}")
        
        if features is None:
            # Cache Miss - Load Audio & Compute
            audio_path = self.get_audio_path(track_id)
            if not os.path.exists(audio_path):
                # Fallback for missing files in dev
                # Generate dummy silence
                y = np.zeros(int(config.SAMPLE_RATE * config.SEGMENT_DURATION * 10)) # 30s silence
            else:
                y = load_audio(audio_path, sr=config.SAMPLE_RATE)
            
            # Process (Segment)
            segments = self.processor.process_track(y)
            if len(segments) == 0:
                segments = np.zeros((1, int(config.SAMPLE_RATE * config.SEGMENT_DURATION)))

            # Compute Features (Mel Spectrograms)
            # Batch compute
            batch_mels = []
            for seg in segments:
                # Normalize
                seg = self.processor.normalize(seg)
                # Compute Mel
                mel = self.feature_extractor.compute_mel_spectrogram(seg)
                batch_mels.append(mel)
            
            features = np.array(batch_mels) # (N, Freq, Time)
            
            # Save to Cache
            if self.cache:
                np.save(cache_path, features)
        
        # Handle Mode return
        if self.mode == 'segment':
            if len(features) > 0:
                # Pick random segment
                choice = np.random.randint(len(features))
                y_out = features[choice]
                # Add channel dimension: (1, F, T)
                y_out = y_out[np.newaxis, ...] 
            else:
                 y_out = np.zeros((1, 128, 130)) # Dummy
        else:
            # Return all segments
            # (N, 1, F, T)
            y_out = features[:, np.newaxis, ...]
        
        return {
            'audio': y_out, # Now this is Spectrogram! Caller must know.
            'label': label,
            'track_id': track_id
        }

def load_tracks(metadata_dir):
    """Simple utility to load tracks without Dataset wrapper"""
    tracks_path = os.path.join(metadata_dir, 'tracks.csv')
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1, 2])
    return tracks
