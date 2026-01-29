from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import numpy as np
import pickle
import os

class GenreLabelEncoder:
    """
    Wrapper for label encoding.
    Supports single-label (LabelEncoder) and multi-label (MultiLabelBinarizer).
    """
    def __init__(self, multi_label=False):
        self.multi_label = multi_label
        if multi_label:
            self.encoder = MultiLabelBinarizer()
        else:
            self.encoder = LabelEncoder()
            
    def fit(self, labels):
        """
        Fit the encoder.
        Args:
            labels: List of labels (or list of lists for multi-label)
        """
        self.encoder.fit(labels)
        
    def transform(self, labels):
        return self.encoder.transform(labels)
        
    def fit_transform(self, labels):
        return self.encoder.fit_transform(labels)
        
    def inverse_transform(self, vectors):
        return self.encoder.inverse_transform(vectors)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.encoder, f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            self.encoder = pickle.load(f)
