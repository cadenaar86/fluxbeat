import unittest
import torch
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.cnn import BaselineCNN
from src.utils.explain import GradCAM

class TestExplainability(unittest.TestCase):
    def setUp(self):
        self.model = BaselineCNN(n_classes=5)
        self.dummy_input = torch.randn(1, 1, 128, 130)
        # Target the last conv layer
        self.cam = GradCAM(self.model, self.model.conv4)

    def test_heatmap_generation(self):
        # Run Grad-CAM
        heatmap = self.cam.generate_cam(self.dummy_input, target_class_idx=0)
        
        # Check shape matches input H, W
        self.assertEqual(heatmap.shape, (128, 130))
        
        # Check normalization (max should be 1.0 if there is any activation)
        self.assertTrue(np.max(heatmap) <= 1.0001)
        self.assertTrue(np.min(heatmap) >= 0.0)

    def tearDown(self):
        self.cam.remove_hooks()

if __name__ == '__main__':
    unittest.main()
