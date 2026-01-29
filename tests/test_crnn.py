import unittest
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.crnn import CRNN

class TestCRNN(unittest.TestCase):
    def setUp(self):
        self.n_classes = 8
        self.model = CRNN(n_classes=self.n_classes)
        # Dummy Mel Spectrogram: (Batch=2, Channels=1, Mels=128, Time=256)
        # Longer time to test subsampling
        self.dummy_input = torch.randn(2, 1, 128, 256)

    def test_forward_pass_dims(self):
        output = self.model(self.dummy_input)
        # Expected output: (Batch, n_classes)
        self.assertEqual(output.shape, (2, self.n_classes))

    def test_variable_length_handling(self):
        # Time=300
        input_var = torch.randn(2, 1, 128, 300)
        output = self.model(input_var)
        self.assertEqual(output.shape, (2, self.n_classes))

if __name__ == '__main__':
    unittest.main()
