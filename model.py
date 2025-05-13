import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F

class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.melspec = T.MelSpectrogram(sample_rate=16000, n_mels=64)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  # Added Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Added Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((14, 14))  # Ensure fixed output size

        # The input size to the first Linear layer is determined by the output of 
        # self.conv (32 channels) and self.pool (14x14 feature map).
        # So, 32 * 14 * 14 = 6272.
        self.fc = nn.Sequential(
            nn.Dropout(0.5),  # Added Dropout for regularization
            nn.Linear(32 * 14 * 14, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # Added Dropout for regularization
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.melspec(x.squeeze(1))  # Remove the 1 channel if needed → [B, N] → [B, 64, T]
        x = x.unsqueeze(1)              # → [B, 1, 64, T]
        x = self.conv(x)                # → [B, 32, H_conv, W_conv]
        x = self.pool(x)                # → [B, 32, 14, 14] (This will be the size before flattening)
        x = x.view(x.size(0), -1)       # Flatten
        return self.fc(x)
