import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch

def compute_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold=0.5):
    """
    Compute classification metrics for multi-label tasks.
    
    Args:
        y_true: Binary ground truth (N, C)
        y_pred_probs: Probability predictions (N, C)
        threshold: Decision threshold
    
    Returns:
        Dictionary of metrics.
    """
    # Binarize predictions
    y_pred = (y_pred_probs > threshold).astype(int)
    
    metrics = {
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred) # Exact match ratio
    }
    
    return metrics

def logits_to_probs(logits):
    """Convert logits to probabilities via sigmoid"""
    return torch.sigmoid(logits).detach().cpu().numpy()
