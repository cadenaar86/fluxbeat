import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z1, z2):
        """
        z1, z2: (Batch, Proj_Dim)
        """
        batch_size = z1.shape[0]
        
        # Concatenate: (2N, Proj_Dim)
        z = torch.cat([z1, z2], dim=0)
        
        # Normalize vectors
        z = F.normalize(z, dim=1)
        
        # Similarity matrix: (2N, 2N)
        # sim[i, j] = z[i] . z[j]
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix.masked_fill_(mask, -9e15)
        
        # Positive pairs: (i, i+N) and (i+N, i)
        # Construct target labels for CrossEntropy
        # For index i (0 to N-1), positive is i+N
        # For index i+N (N to 2N-1), positive is i
        
        # Alternative efficient implementation:
        # We want to classify the positive pair among 2(N-1) negatives.
        
        labels = torch.cat([
            torch.arange(batch_size, 2*batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device)
        ], dim=0)
        
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

class MultiTaskLoss(nn.Module):
    """
    Combined loss for Genre (Classification) and Tempo (Regression).
    """
    def __init__(self, tempo_weight=1.0):
        super().__init__()
        self.tempo_weight = tempo_weight
        self.genre_criterion = nn.BCEWithLogitsLoss()
        self.tempo_criterion = nn.MSELoss()
        
    def forward(self, outputs, targets):
        """
        outputs: dict {'genre': logits, 'tempo': pred_bpm}
        targets: dict {'genre': labels, 'tempo': true_bpm}
        """
        genre_loss = self.genre_criterion(outputs['genre'], targets['genre'])
        tempo_loss = self.tempo_criterion(outputs['tempo'], targets['tempo'])
        
        total_loss = genre_loss + (self.tempo_weight * tempo_loss)
        return total_loss, {'genre_loss': genre_loss, 'tempo_loss': tempo_loss}
