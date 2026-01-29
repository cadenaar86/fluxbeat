import torch
import os
import argparse
from ..models.cnn import BaselineCNN
from ..models.crnn import CRNN
from ..models.transformer import AudioTransformer
from .. import config

def export_model(model_name, output_path, n_classes=10):
    device = torch.device('cpu')
    
    # Factory
    if model_name == 'BaselineCNN':
        model = BaselineCNN(n_classes=n_classes)
    elif model_name == 'CRNN':
        model = CRNN(n_classes=n_classes)
    elif model_name == 'AudioTransformer':
        model = AudioTransformer(n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    model.eval()
    
    # Dummy Input
    # (Batch, 1, Freq, Time)
    dummy_input = torch.randn(1, 1, 128, 130)
    
    try:
        # Trace
        traced_script_module = torch.jit.trace(model, dummy_input)
        traced_script_module.save(output_path)
        print(f"Successfully exported {model_name} to {output_path}")
        print(f"Input shape used: {dummy_input.shape}")
    except Exception as e:
        print(f"Failed to export {model_name}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='BaselineCNN', help='Model to export')
    parser.add_argument('--out', type=str, default='model.pt', help='Output path')
    args = parser.parse_args()
    
    export_model(args.model, args.out)
