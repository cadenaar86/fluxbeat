import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference.engine import InferenceEngine
from src import config

class TestInference(unittest.TestCase):
    def setUp(self):
        # Initialize without loading weights (random weights)
        self.engine = InferenceEngine()
        # Mock label encoder for consistent keys
        # We can't easily mock the label encoder inside without DI, 
        # so we will check generic keys "Class_0"

    def test_prediction_output(self):
        # 10 seconds of random audio
        audio = np.random.rand(config.SAMPLE_RATE * 10)
        
        results = self.engine.predict_audio(audio)
        
        # Should be a dict
        self.assertIsInstance(results, dict)
        
        # Check values are probabilities 0-1
        for key, val in results.items():
            self.assertTrue(0.0 <= val <= 1.0)
            
        # Check we have 10 classes (default fallback)
        self.assertEqual(len(results), 10)

if __name__ == '__main__':
    unittest.main()
