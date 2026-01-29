import unittest
import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.cnn import BaselineCNN
from src.models.train import Trainer
from torch.utils.data import DataLoader, TensorDataset

class TestModel(unittest.TestCase):
    def setUp(self):
        self.n_classes = 5
        self.model = BaselineCNN(n_classes=self.n_classes)
        # Dummy Mel Spectrogram: (Batch=4, Channels=1, Mels=128, Time=130)
        self.dummy_input = torch.randn(4, 1, 128, 130)

    def test_forward_pass(self):
        output = self.model(self.dummy_input)
        # Expected output shape: (Batch, n_classes)
        self.assertEqual(output.shape, (4, self.n_classes))

    def test_training_step(self):
        # Create dummy dataset
        # Inputs: (10 samples, 128, 130) - Trainer adds channel dim usually or expects it.
        # Trainer expects dictionary batch.
        inputs = torch.randn(10, 128, 130) # Missing channel, trainer handles it
        labels = torch.randint(0, 2, (10, self.n_classes)).float() # Multi-label binary
        
        dataset = TensorDataset(inputs, labels)
        
        # Wrap in expected dict format for custom Trainer
        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, tensor_ds):
                self.ds = tensor_ds
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                x, y = self.ds[idx]
                return {'audio': x, 'label': y}
                
        loader = DataLoader(DictDataset(dataset), batch_size=2)
        
        trainer = Trainer(self.model, loader, loader, device='cpu')
        
        # Run one epoch
        metrics = trainer.train_epoch()
        
        self.assertIn('loss', metrics)
        self.assertIn('f1_macro', metrics)
        print(f"Training check metrics: {metrics}")

if __name__ == '__main__':
    unittest.main()
