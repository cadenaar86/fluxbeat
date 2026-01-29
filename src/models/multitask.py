import torch
import torch.nn as nn

class MultiTaskModel(nn.Module):
    """
    Multi-Task Learning Model.
    Shared Encoder -> Genre Head (Classification) + Tempo Head (Regression)
    """
    def __init__(self, encoder, embed_dim=256, n_genres=10):
        """
        Args:
            encoder: A model that outputs embeddings (B, embed_dim)
            embed_dim: Dimension of encoder output
            n_genres: Number of genre classes
        """
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        
        # Genre Head
        self.genre_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embed_dim, n_genres)
        )
        
        # Tempo Head
        # Predicts BPM (Beats Per Minute)
        self.tempo_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Single scalar output
        )
        
    def forward(self, x):
        # We assume encoder returns just the embedding or tuple
        # If encoder returns tuple (like SimCLR wrapper or some ViTs), handle it.
        # Our BaselineCNN returns logits (unusable here unless we modify it to return features)
        # or we wrap a feature extractor.
        
        # Let's assume 'encoder' returns the shared representation `h`
        features = self.encoder(x)
        
        # Genre Logits
        genre_logits = self.genre_head(features)
        
        # Tempo Prediction (BPM)
        tempo_pred = self.tempo_head(features)
        
        return {
            'genre': genre_logits,
            'tempo': tempo_pred
        }
