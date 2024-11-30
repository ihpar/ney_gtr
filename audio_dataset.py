import librosa
import numpy as np
import seaborn as sns
import librosa.display
from constants import *
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    def __init__(self, file_path, part):
        super().__init__()
        self.file_path = file_path
        self.part = part
        self.X = None
        self.Y = None

    def _get_features(self, file_path):
        signal, _ = librosa.load(file_path, mono=True, sr=SR)
        offset = WINDOW_SAMPLE_LEN - OVERLAP
        num_windows = np.ceil(
            (len(signal) - WINDOW_SAMPLE_LEN) / (WINDOW_SAMPLE_LEN - OVERLAP) + 1)
        num_windows = int(np.ceil(num_windows / 4) * 4)
        for i in range(num_windows):
            start = i * offset
            end = start + WINDOW_SAMPLE_LEN
            sig_window = signal[start:end]
            if len(sig_window) < WINDOW_SAMPLE_LEN:
                sig_window = np.pad(
                    sig_window, (0, WINDOW_SAMPLE_LEN - len(sig_window)), mode="constant")

            stft = librosa.stft(sig_window, n_fft=N_FFT, hop_length=HOP)[:-1]
            magnitude = np.abs(stft)
            magnitude = np.expand_dims(magnitude, axis=0)
            phase = np.angle(stft)
            phase = np.expand_dims(phase, axis=0)
            db = librosa.amplitude_to_db(magnitude)
            db = np.expand_dims(db, axis=0)

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


if __name__ == "__main__":
    dataset = AudioDataset("dataset/test/test_0.wav")
