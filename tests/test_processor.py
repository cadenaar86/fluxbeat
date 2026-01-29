import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.processor import AudioProcessor
from src import config

class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = AudioProcessor(sample_rate=1000, segment_duration=2)
        # Expected samples per segment = 1000 * 2 = 2000

    def test_segmentation_exact_fit(self):
        # Create 3 segments worth of data (6000 samples)
        audio = np.random.rand(6000)
        segments = self.processor.process_track(audio)
        self.assertEqual(segments.shape, (3, 2000))

    def test_segmentation_remainder(self):
        # Create 3.5 segments worth of data
        audio = np.random.rand(7500)
        segments = self.processor.process_track(audio)
        # Should truncate the remainder
        self.assertEqual(segments.shape, (3, 2000))

    def test_segmentation_padding(self):
        # Create 0.5 segments worth of data
        audio = np.random.rand(1000)
        segments = self.processor.process_track(audio)
        # Should pad to 1 segment
        self.assertEqual(segments.shape, (1, 2000))
        # Check padding (last 1000 samples should be 0 - assuming default 0 pad)
        self.assertTrue(np.all(segments[0, 1000:] == 0))

    def test_normalization(self):
        audio = np.array([0.5, 1.0, 0.5])
        norm = self.processor.normalize(audio)
        self.assertEqual(np.max(np.abs(norm)), 1.0)
        
        audio_quiet = np.array([0.1, 0.2, 0.1])
        norm_quiet = self.processor.normalize(audio_quiet)
        self.assertEqual(np.max(np.abs(norm_quiet)), 1.0)

if __name__ == '__main__':
    unittest.main()
