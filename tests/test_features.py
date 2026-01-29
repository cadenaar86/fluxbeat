import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.features import FeatureExtractor
from src import config

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor()
        # Generate 3 seconds of random audio
        self.audio_len = int(config.SAMPLE_RATE * 3)
        self.audio = np.random.rand(self.audio_len)

    def test_mel_spectrogram_shape(self):
        mel = self.extractor.compute_mel_spectrogram(self.audio)
        # Expected shape: (n_mels, time_steps)
        # time_steps = ceil(dataset_len / hop_length)
        # 22050 * 3 / 512 approx 129 + 1 = 130
        expected_time_steps = int(np.ceil(self.audio_len / config.HOP_LENGTH))
        
        self.assertEqual(mel.shape[0], config.N_MELS)
        # Allow +/- 1 frame difference due to padding logic in librosa
        self.assertTrue(abs(mel.shape[1] - expected_time_steps) <= 1)

    def test_mfcc_shape(self):
        n_mfcc = 13
        mfcc = self.extractor.compute_mfcc(self.audio, n_mfcc=n_mfcc)
        self.assertEqual(mfcc.shape[0], n_mfcc)
        self.assertGreater(mfcc.shape[1], 0)

    def test_chroma_shape(self):
        chroma = self.extractor.compute_chroma(self.audio)
        self.assertEqual(chroma.shape[0], 12) # 12 semitones
        self.assertGreater(chroma.shape[1], 0)

if __name__ == '__main__':
    unittest.main()
