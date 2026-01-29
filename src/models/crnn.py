import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    """
    CRNN: Convolutional Recurrent Neural Network.
    Combines CNN for feature extraction and LSTM for temporal modeling.
    """
    def __init__(self, n_input_channels=1, n_classes=10, lstm_hidden_size=128):
        super(CRNN, self).__init__()
        
        # 1. CNN Feature Extractor
        # We want to downsample Frequency mostly, and Time partially.
        # Input: (B, 1, 128, T)
        
        self.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 2)) # (64, T/2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((2, 2)) # (32, T/4)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d((2, 2)) # (16, T/8)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d((2, 2)) # (8, T/16)
        
        # At this point, Frequency dim is 128 / 16 = 8.
        # Channels = 512.
        # We can crush Frequency dim to 1 via another pool or learn it.
        # Let's pool Frequency to 1.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None)) 
        # Output: (B, 512, 1, T/16) -> Squeeze -> (B, 512, T')
        
        # 2. Recurrent Block
        self.lstm = nn.LSTM(
            input_size=512, 
            hidden_size=lstm_hidden_size, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.5
        )
        
        # 3. Classifier
        self.dropout = nn.Dropout(0.5)
        # Bidirectional = 2 * hidden_size
        self.fc = nn.Linear(lstm_hidden_size * 2, n_classes)
        
    def forward(self, x):
        # x: (B, 1, F, T)
        
        # CNN
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # (B, 512, F', T') -> (B, 512, 1, T')
        x = self.adaptive_pool(x)
        x = x.squeeze(2) # (B, 512, T')
        
        # Permute for LSTM: (B, T', Features)
        x = x.permute(0, 2, 1)
        
        # RNN
        self.lstm.flatten_parameters() # efficient for GPU
        x, (hn, cn) = self.lstm(x)
        # x: (B, T', 2*Hidden)
        
        # Aggregation: Use the last step? Or Average?
        # A common trick is Global Avg Pooling over time for the RNN output too.
        # Or just take the last state. 
        # Since it's bidirectional, we can concatenate forward_last and backward_last.
        # Or mean over time.
        # Let's use Mean over Time (captures "vibe" across whole duration)
        x = x.mean(dim=1)
        
        # FC
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
