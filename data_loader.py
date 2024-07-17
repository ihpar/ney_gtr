import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

NEY_SPECTROGRAM_DIR = "dataset/spectrograms/ney/"
GTR_SPECTROGRAM_DIR = "dataset/spectrograms/ac_gtr/"


class SpectrogramDataset(Dataset):
    def __init__(self,
                 x_files,
                 y_files,
                 ney_spectrogram_dir,
                 gtr_spectrogram_dir):
        self.x_files = x_files
        self.y_files = y_files
        self.ney_spectrogram_dir = ney_spectrogram_dir
        self.gtr_spectrogram_dir = gtr_spectrogram_dir
        self.x = []
        self.y = []

        # gtr files
        for f in self.x_files:
            spectrogram = np.load(
                self.gtr_spectrogram_dir + f, allow_pickle=True)
            self.x.append(spectrogram)

        # ney files
        for f in self.y_files:
            spectrogram = np.load(
                self.ney_spectrogram_dir + f, allow_pickle=True)
            self.y.append(spectrogram)

        # prepare gtr spectrograms
        self.x = np.array(self.x)
        # min max scaling
        self.x = (self.x - np.min(self.x)) / (np.max(self.x) - np.min(self.x))
        # expand dims to imitate grayscale img format
        self.x = np.expand_dims(self.x, axis=3)

        # prepare ney spectrograms
        self.y = np.array(self.y)
        # min max scaling
        self.y = (self.y - np.min(self.y)) / (np.max(self.y) - np.min(self.y))
        # expand dims to imitate grayscale img format
        self.y = np.expand_dims(self.y, axis=3)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


if __name__ == "__main__":
    with open("dataset/dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
        x_files = dataset["gtr"]
        y_files = dataset["ney"]

    x_train_files, x_test_files, y_train_files, y_test_files = train_test_split(
        x_files, y_files, test_size=0.2, random_state=42)

    train_dataset = SpectrogramDataset(x_train_files,
                                       y_train_files,
                                       NEY_SPECTROGRAM_DIR,
                                       GTR_SPECTROGRAM_DIR)
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=True)

    x, y = next(iter(train_data_loader))
    print(x.shape)
