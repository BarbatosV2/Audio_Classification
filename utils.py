import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as T

LABELS = ['cat', 'dog']  # Will be populated in dataset

def preprocess_audio(file_path, sample_rate=16000):
    waveform, sr = torchaudio.load(file_path, normalize=True)

    # Resample if needed
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Apply MelSpectrogram transform
    mel_spec = T.MelSpectrogram(sample_rate=sample_rate, n_mels=64)(waveform)

    # Normalize the spectrogram
    mel_spec = mel_spec / mel_spec.max()  # Normalize between 0 and 1

    return mel_spec

class AudioDataset(Dataset):
    def __init__(self, audio_dir, csv_file, fixed_length=16000):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        global LABELS
        LABELS = sorted(self.data['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
        self.num_classes = len(self.label_to_idx)
        self.fixed_length = fixed_length  # Number of samples to pad or trim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.data.iloc[idx, 0])
        label_str = self.data.iloc[idx, 1]
        label = self.label_to_idx[label_str]

        waveform, sr = torchaudio.load(audio_path)  # waveform: [1, N]
        waveform = waveform.mean(0)  # Convert to mono: [N]

        # Pad or trim to fixed length
        if waveform.size(0) < self.fixed_length:
            pad_len = self.fixed_length - waveform.size(0)
            padding = torch.zeros(pad_len)
            waveform = torch.cat([waveform, padding])
        else:
            waveform = waveform[:self.fixed_length]

        return waveform, label


def collate_fn(batch):
    waveforms, labels = zip(*batch)
    waveforms = [w.squeeze(0) for w in waveforms]  # remove unnecessary channel
    waveforms = torch.stack(waveforms)  # [B, N]
    waveforms = waveforms.unsqueeze(1)  # [B, 1, N]
    labels = torch.tensor(labels)
    return waveforms, labels

def get_device(device_index=0):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_index}")
    else:
        return torch.device("cpu")
