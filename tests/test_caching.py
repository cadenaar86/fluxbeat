import unittest
import torch
import numpy as np
import sys
import os
import shutil
import pandas as pd
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.data.loader import FMADataset

class TestCaching(unittest.TestCase):
    def setUp(self):
        # Mock metadata
        self.mock_tracks = pd.DataFrame({
            ('set', 'subset', ''): ['small'],
            ('set', 'split', ''): ['train'],
            ('track', 'genre_top', ''): ['Rock']
        }, index=[123456]) # Track ID
        
        # Mock paths
        self.original_metadata_dir = config.METADATA_DIR
        self.original_cache_dir = config.CACHE_DIR
        
        # Use a temp dir for cache test
        self.test_cache_dir = os.path.join(config.PROCESSED_DATA_DIR, 'test_cache')
        config.CACHE_DIR = self.test_cache_dir
        os.makedirs(self.test_cache_dir, exist_ok=True)
        
        # Patch FMADataset._load_metadata to return mock
        self.patcher = patch.object(FMADataset, '_load_metadata', return_value=self.mock_tracks)
        self.mock_load = self.patcher.start()
        
        # Patch load_audio (so we don't need real mp3)
        self.audio_patcher = patch('src.data.loader.load_audio', return_value=np.random.rand(22050*10))
        self.mock_audio = self.audio_patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.audio_patcher.stop()
        config.CACHE_DIR = self.original_cache_dir
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)

    def test_cache_creation(self):
        # Initialize dataset with cache=True
        dataset = FMADataset(cache=True)
        
        # Access item 0 (track 123456)
        # Should TRIGGER computation and save
        item = dataset[0]
        
        # Check output is spectrogram (1, F, T)
        self.assertEqual(item['audio'].shape, (1, 128, 130))
        
        # Check file exists
        expected_path = os.path.join(self.test_cache_dir, 'small', '123456.npy')
        self.assertTrue(os.path.exists(expected_path))
        
        # Access item again
        # Should LOAD from cache (you'd need to mock computation being called to verify strictly, 
        # but basic existence check is good)
        item2 = dataset[0]
        self.assertEqual(item2['audio'].shape, (1, 128, 130))

if __name__ == '__main__':
    unittest.main()
