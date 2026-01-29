import unittest
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.multitask import MultiTaskModel
from src.models.losses import MultiTaskLoss

class MockFeatureEncoder(torch.nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        # Map input (Flattened spectrogram) to embed_dim
        self.linear = torch.nn.Linear(128*130, embed_dim) 
        
    def forward(self, x):
        # x: (B, 1, 128, 130)
        x = x.view(x.size(0), -1)
        return self.linear(x)

class TestMultiTask(unittest.TestCase):
    def setUp(self):
        self.encoder = MockFeatureEncoder()
        self.model = MultiTaskModel(self.encoder, embed_dim=256, n_genres=5)
        self.loss_fn = MultiTaskLoss(tempo_weight=0.5)
        
        # Input
        self.dummy_input = torch.randn(2, 1, 128, 130)

    def test_forward_pass(self):
        out = self.model(self.dummy_input)
        self.assertIn('genre', out)
        self.assertIn('tempo', out)
        
        # Check shapes
        self.assertEqual(out['genre'].shape, (2, 5))
        self.assertEqual(out['tempo'].shape, (2, 1))

    def test_loss_computation(self):
        outputs = self.model(self.dummy_input)
        
        targets = {
            'genre': torch.randint(0, 2, (2, 5)).float(), # Multi-label
            'tempo': torch.tensor([[120.0], [90.0]]).float()
        }
        
        total_loss, details = self.loss_fn(outputs, targets)
        
        self.assertTrue(torch.isfinite(total_loss))
        self.assertIn('genre_loss', details)
        self.assertIn('tempo_loss', details)

if __name__ == '__main__':
    unittest.main()
