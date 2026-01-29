import torch
import numpy as np
import os
from .. import config
from ..models.cnn import BaselineCNN
from ..data.processor import AudioProcessor
from ..utils.audio import load_audio
from ..utils.labels import GenreLabelEncoder
from ..data.features import FeatureExtractor
from ..utils.metrics import logits_to_probs

class InferenceEngine:
    """
    Real-time inference engine for FluxBeat.
    """
    def __init__(self, model_path=None, labels_path=None, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Labels
        self.label_encoder = None
        if labels_path and os.path.exists(labels_path):
            self.label_encoder = GenreLabelEncoder()
            self.label_encoder.load(labels_path)
            self.n_classes = len(self.label_encoder.encoder.classes_)
        else:
            # Fallback / Dummy
            self.n_classes = 10 
            print("Warning: No labels provided, assuming 10 classes.")

        # Load Model
        self.model = BaselineCNN(n_classes=self.n_classes)
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded model from {model_path}")
        else:
            print("Warning: No model path provided or file missing. Using uninitialized model.")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Components
        self.processor = AudioProcessor()
        self.feature_extractor = FeatureExtractor()

    def predict_audio(self, audio_array, overlap=0.5):
        """
        Predict genres for raw audio array.
        """
        # 1. Segment (Sliding Window)
        segments = self.processor.segment_sliding_window(audio_array, overlap=overlap)
        
        # 2. Extract Features
        # Batch processing
        batch_mels = []
        for segment in segments:
            # Normalize
            segment = self.processor.normalize(segment)
            # Mel Spec
            mel = self.feature_extractor.compute_mel_spectrogram(segment)
            batch_mels.append(mel)
            
        # Stack -> (N, Mels, Time) -> (N, 1, Mels, Time)
        batch_input = np.array(batch_mels)
        batch_input = torch.tensor(batch_input).unsqueeze(1).float().to(self.device)
        
        # 3. Predict
        with torch.no_grad():
            logits = self.model(batch_input)
            probs = logits_to_probs(logits) # (N, n_classes)
            
        # 4. Aggregate (Mean pooling across time windows)
        avg_probs = np.mean(probs, axis=0) # (n_classes)
        
        # 5. Format Output
        results = {}
        if self.label_encoder:
            classes = self.label_encoder.encoder.classes_
            for idx, cls in enumerate(classes):
                results[cls] = float(avg_probs[idx])
        else:
            for idx, prob in enumerate(avg_probs):
                results[f"Class_{idx}"] = float(prob)
                
        return results

    def predict_file(self, file_path, overlap=0.5):
        audio = load_audio(file_path)
        if len(audio) == 0:
            return None
        return self.predict_audio(audio, overlap=overlap)
