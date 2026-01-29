import torch
import time
import numpy as np
from ..models.cnn import BaselineCNN
from ..models.crnn import CRNN
from ..models.transformer import AudioTransformer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_model(model, input_shape=(1, 1, 128, 130), device='cpu', n_loops=50):
    model.eval()
    model.to(device)
    x = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
            
    # Timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_loops):
            _ = model(x)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_loops
    params = count_parameters(model)
    
    return avg_time * 1000, params # ms, count

def run_benchmarks():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device}...")
    
    models = {
        'BaselineCNN': BaselineCNN(n_classes=10),
        'CRNN': CRNN(n_classes=10),
        'AudioTransformer': AudioTransformer(n_classes=10)
    }
    
    # Input shape: (Batch, Channel, Freq, Time)
    # 3 seconds of audio at sr=22050 -> ~130 frames for Mel Spectrogram
    input_shape = (1, 1, 128, 130) 
    
    print(f"{'Model':<20} | {'Params':<12} | {'Latency (ms)':<12}")
    print("-" * 50)
    
    for name, model in models.items():
        try:
            latency, params = benchmark_model(model, input_shape, device)
            print(f"{name:<20} | {params:<12,} | {latency:.2f}")
        except Exception as e:
            print(f"{name:<20} | Error: {e}")

if __name__ == '__main__':
    run_benchmarks()
