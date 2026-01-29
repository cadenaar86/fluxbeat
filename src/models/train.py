import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
from ..utils.metrics import compute_metrics, logits_to_probs

class Trainer:
    """
    Handling training, validation, and checkpointing.
    """
    def __init__(self, model, train_loader, val_loader, 
                 criterion=None, optimizer=None, device=None,
                 checkpoint_dir='checkpoints'):
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Default Multi-Label Loss
        self.criterion = criterion if criterion else nn.BCEWithLogitsLoss()
        
        self.optimizer = optimizer if optimizer else optim.Adam(model.parameters(), lr=1e-3)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        # We also want to track metrics on train set roughly
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Unpack batch.
            # Assuming batch is dict {'audio': ..., 'label': ...} from loader
            # But wait, loader returns audio. Features might be computed on the fly or pre-computed.
            # Let's assume input is already Spectrogram or we transform it here.
            # For this baseline implementation, let's assume the loader yields
            # features ready for the model (or we add a transform in loader).
            # We'll adjust based on verifying loader.py behavior.
            
            inputs = batch['audio'].to(self.device)
            labels = batch['label'].float().to(self.device) # Float for BCE
            
            # Add channel dim if missing (N, H, W) -> (N, 1, H, W)
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # Store for metrics
            all_preds.append(logits_to_probs(outputs))
            all_labels.append(labels.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(self.train_loader.dataset)
        
        # Compute train metrics
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        train_metrics = compute_metrics(all_labels, all_preds)
        train_metrics['loss'] = epoch_loss
        
        return train_metrics

    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                inputs = batch['audio'].to(self.device)
                labels = batch['label'].float().to(self.device)
                
                if inputs.dim() == 3:
                    inputs = inputs.unsqueeze(1)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                
                all_preds.append(logits_to_probs(outputs))
                all_labels.append(labels.detach().cpu().numpy())
                
        epoch_loss = running_loss / len(self.val_loader.dataset)
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        val_metrics = compute_metrics(all_labels, all_preds)
        val_metrics['loss'] = epoch_loss
        
        return val_metrics

    def save_checkpoint(self, name):
        path = os.path.join(self.checkpoint_dir, name)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
