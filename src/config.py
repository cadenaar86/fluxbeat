import os
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

# --- Audio settings ---
SAMPLE_RATE = 22050
DURATION = 30  # Duration of full tracks in FMA is usually 30s
SEGMENT_DURATION = 3  # Input duration for the model
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
CACHE_DIR = os.path.join(PROCESSED_DATA_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# --- Transformer Config (ViT) ---
PATCH_SIZE = (16, 16) # (Freq, Time)
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 4
DROPOUT = 0.1
