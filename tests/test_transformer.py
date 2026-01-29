import unittest
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.transformer import AudioTransformer

class TestAudioTransformer(unittest.TestCase):
    def setUp(self):
        self.n_classes = 5
        self.img_size = (128, 128)
        self.model = AudioTransformer(
            img_size=self.img_size, 
            patch_size=(16, 16), 
            n_classes=self.n_classes,
            embed_dim=64, # small for test
            depth=2,
            num_heads=2
        )
        self.dummy_input = torch.randn(2, 1, 128, 128) # Perfectly divisible

    def test_forward_pass(self):
        output = self.model(self.dummy_input)
        # Expected: (B, n_classes)
        self.assertEqual(output.shape, (2, self.n_classes))

    def test_patch_embedding_shape(self):
        # 128x128 / 16x16 = 8x8 = 64 patches
        x = self.model.patch_embed(self.dummy_input)
        self.assertEqual(x.shape, (2, 64, 64)) # (B, N_Patches, Embed_Dim)

    def test_slightly_off_size(self):
        # 128 x 130
        model = AudioTransformer(img_size=(128, 130))
        input_var = torch.randn(2, 1, 128, 130)
        # Conv2d striding should ignore last 2 pixels of W
        output = model(input_var)
        self.assertEqual(output.shape, (2, 10))

if __name__ == '__main__':
    unittest.main()
