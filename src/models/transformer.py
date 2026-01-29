import torch
import torch.nn as nn
import numpy as np
import math

class PatchEmbed(nn.Module):
    """
    Split 2D Spectrogram Image into Patches and then flatten them.
    Input: (B, C, H, W)
    Output: (B, Num_Patches, Embed_Dim)
    """
    def __init__(self, img_size=(128, 128), patch_size=(16, 16), in_chans=1, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        
        # We can use a Conv2d with kernel_size=patch_size and stride=patch_size
        # to effectively patch and project in one step.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        # Check dimensions? Or assume standard sized segments.
        # Ideally, we crop or pad if H/W not divisible.
        
        x = self.proj(x) # (B, Embed, H', W')
        # Flatten H', W' -> Sequence
        x = x.flatten(2) # (B, Embed, N_Patches)
        x = x.transpose(1, 2) # (B, N_Patches, Embed)
        return x

class PositionalEncoding(nn.Module):
    """
    Learnable Positional Encoding.
    """
    def __init__(self, num_patches, embed_dim, dropout=0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) # +1 for CLS
        self.dropout = nn.Dropout(dropout)
        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, N, E)
        # Add pos embed (broadcast batch)
        # If x len matches num_patches + 1 (with CLS)
        if x.size(1) == self.pos_embed.size(1):
            x = x + self.pos_embed
        else:
            # Handle variable length or missing CLS if applicable
            x = x + self.pos_embed[:, :x.size(1), :]
        return self.dropout(x)

class AudioTransformer(nn.Module):
    """
    ViT-style Transformer for Audio Classification.
    """
    def __init__(self, 
                 img_size=(128, 130), 
                 patch_size=(16, 16), 
                 in_chans=1, 
                 n_classes=10, 
                 embed_dim=256, 
                 depth=4, 
                 num_heads=8, 
                 mlp_ratio=4., 
                 dropout=0.1):
        super().__init__()
        
        # Calculate expected number of patches
        # For spectrogram 128x130, and patch 16x16:
        # H patches = 128/16 = 8
        # W patches = 130/16 = 8 (remainder 2 ignored by Conv2d usually)
        # So 8*8 = 64 patches.
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Positional Encoding
        # Note: If input length varies (not exactly img_size), this fixed pos embed is tricky.
        # We assume fixed input size for standard ViT.
        self.pos_drop = nn.Dropout(p=dropout)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            dropout=dropout,
            batch_first=True,
            norm_first=True # Pre-Norm usually better
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classifier Head
        self.norm_head = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        
    def forward(self, x):
        # x: (B, 1, H, W)
        
        B = x.shape[0]
        x = self.patch_embed(x) # (B, N, E)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, N+1, E)
        
        # Add Positional Embedding
        # Interpolate pos_embed if size mismatch?
        # For now assume match.
        if x.size(1) == self.pos_embed.size(1):
             x = x + self.pos_embed
        else:
             # Basic interpolation or slicing could go here
             x = x + self.pos_embed[:, :x.size(1), :]
             
        x = self.pos_drop(x)
        
        # Transformer
        x = self.encoder(x)
        
        # Head (Use CLS token)
        cls_out = x[:, 0]
        cls_out = self.norm_head(cls_out)
        logits = self.head(cls_out)
        
        return logits
