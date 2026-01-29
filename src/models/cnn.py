import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    """
    Standard 2D CNN for Audio Classification using Mel Spectrograms.
    Architecture: 4 Conv Blocks -> Global Average Pooling -> FC Layer.
    """
    def __init__(self, n_input_channels=1, n_classes=10):
        super(BaselineCNN, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2) # Reduces dims by 2
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        # Conv Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)

        # Fully Connected Block
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, n_classes)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Channels, Freq(Mels), Time)
        """
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Block 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Global Average Pooling
        # x shape is (Batch, 256, H, W). We want (Batch, 256).
        x = x.mean(dim=[-2, -1]) 
        
        # FC
        x = self.dropout(x)
        x = self.fc(x)
        
        # Notes: We return logits. 
        # Loss function (BCEWithLogitsLoss) will apply Sigmoid + CrossEntropy.
        # Inference will apply Sigmoid manually.
        return x
