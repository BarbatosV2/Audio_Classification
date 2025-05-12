import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F

class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.melspec = T.MelSpectrogram(sample_rate=16000, n_mels=64)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((14, 14))  # Ensure fixed output size

        # You may need to change this input size
        self.fc = nn.Sequential(
            nn.Linear(32 * 14 * 14, 128),  # Adjust this size after debugging
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.melspec(x.squeeze(1))  # Remove the 1 channel if needed → [B, N] → [B, 64, T]
        x = x.unsqueeze(1)              # → [B, 1, 64, T]
        x = self.conv(x)                # → [B, 32, H_conv, W_conv]
        x = self.pool(x)                # → [B, 32, 14, 14] (This will be the size before flattening)
        x = x.view(x.size(0), -1)       # Flatten
        return self.fc(x)
