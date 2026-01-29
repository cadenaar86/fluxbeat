# FluxBeat User Guide 📘

This guide provides detailed instructions on how to use FluxBeat for data preparation, training, evaluation, and deployment.

## 1. Data Preparation

FluxBeat uses the **Free Music Archive (FMA)** dataset.

### Download Data
Use the built-in downloader script to fetch the `fma_small` (8GB) dataset:
```bash
python src/data/downloader.py --size small
```
*Note: Ensure you have `curl` and `unzip` installed.*

### Feature Caching
To speed up training, you can pre-compute Mel Spectrograms:
```python
from src.data.loader import FMADataset
# Instantiating with cache=True triggers computation
ds = FMADataset(subset='small', cache=True)
# Iterate once to fill cache (or let the Trainer do it in the first epoch)
```

## 2. Training Models

### Basic Training
You can train the Baseline CNN using the `Trainer` class. Create a script (e.g., `train_cnn.py`):
```python
from src.models.cnn import BaselineCNN
from src.models.train import Trainer
from src.data.loader import FMADataset
from torch.utils.data import DataLoader

# Data
train_ds = FMADataset(split='train', cache=True)
val_ds = FMADataset(split='validation', cache=True)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# Model
model = BaselineCNN(n_classes=8) # fma_small has 8 genres

# Train
trainer = Trainer(model, train_loader, val_loader)
trainer.train(epochs=10)
```

### Training Advanced Models
To train the **CRNN** or **AudioTransformer**, simply swap the model instantiation:
```python
from src.models.transformer import AudioTransformer
model = AudioTransformer(n_classes=8)
```

## 3. Advanced Features

### Self-Supervised Pretraining (SimCLR)
To pretrain on unlabeled data using Contrastive Learning:
1. Initialize `AudioTransformer` or `BaselineCNN` as the encoder.
2. Wrap it in `SimCLR` (`src/models/ssl.py`).
3. Use the `AudioAugmentor` (`src/data/augment.py`) in your dataloader.
4. Train using `NTXentLoss` (`src/models/losses.py`).

### Multi-Task Learning
To predict Tempo and Genre together:
```python
from src.models.multitask import MultiTaskModel
from src.models.losses import MultiTaskLoss

mtl_model = MultiTaskModel(base_encoder, n_genres=8)
criterion = MultiTaskLoss(tempo_weight=1.0)
```
*Ensure your dataset loader returns 'tempo' in the dictionary.*

## 4. Benchmarking & Export

### Compare Latency
```bash
python -m src.utils.benchmark
```

### Export to TorchScript
Prepare your model for C++ or mobile deployment:
```bash
python -m src.utils.export --model AudioTransformer --out transformer_v1.pt
```

## 5. API Deployment
FluxBeat includes a production-ready FastAPI server.

```bash
uvicorn src.api.server:app --reload
```
- **Endpoint**: `POST /predict`
- **Payload**: `multipart/form-data` with an audio file.
- **Response**: JSON with genre probabilities.
