import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    """
    SimCLR framework.
    """
    def __init__(self, encoder, embed_dim=256, projection_dim=128):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        
        # Projection Head
        # (Embed -> ReLU -> Projection)
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim)
        )
        
    def forward(self, x):
        # x: Input batch
        # Encode
        h = self.encoder(x)
        
        # SimCLR Projection
        z = self.projection_head(h)
        return h, z

class SimCLRWrapper(nn.Module):
    """
    Wrapper to handle the two-view logic internally if needed,
    or just facilitate the forward pass.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x1, x2):
        _, z1 = self.model(x1)
        _, z2 = self.model(x2)
        return z1, z2
