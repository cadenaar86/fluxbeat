import unittest
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.augment import AudioAugmentor
from src.models.ssl import SimCLR
from src.models.losses import NTXentLoss

class MockEncoder(torch.nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.linear = torch.nn.Linear(10, output_dim)
        
    def forward(self, x):
        # Flatten dummy input
        x = x.view(x.size(0), -1)
        return self.linear(x[:, :10])

class TestSSL(unittest.TestCase):
    def setUp(self):
        self.augmentor = AudioAugmentor()
        self.encoder = MockEncoder()
        self.model = SimCLR(self.encoder, embed_dim=256, projection_dim=128)
        self.loss_fn = NTXentLoss()

    def test_augmentation(self):
        # Dummy Spec: (1, 128, 128)
        spec = torch.randn(1, 128, 128)
        out = self.augmentor(spec)
        self.assertEqual(out.shape, spec.shape)
        
    def test_simclr_forward(self):
        dummy_input = torch.randn(2, 1, 128, 128)
        h, z = self.model(dummy_input)
        self.assertEqual(z.shape, (2, 128)) # Projection dim
        
    def test_loss(self):
        z1 = torch.randn(4, 128)
        z2 = torch.randn(4, 128)
        loss = self.loss_fn(z1, z2)
        self.assertTrue(torch.isfinite(loss))

if __name__ == '__main__':
    unittest.main()
