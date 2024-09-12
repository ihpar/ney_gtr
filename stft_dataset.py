import pickle
import numpy as np
from constants import *
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

ney_feature_dirs = sorted(
    [f for f in Path(NEY_FEATURE_DIR).iterdir() if f.is_dir()])
gtr_feature_dirs = sorted(
    [f for f in Path(GTR_FEATURE_DIR).iterdir() if f.is_dir()])

x_train_dirs, x_test_dirs, y_train_dirs, y_test_dirs = train_test_split(
    gtr_feature_dirs, ney_feature_dirs, test_size=0.2, random_state=42)

# print(len(x_train_dirs), len(x_test_dirs))
# print(x_train_dirs[0], y_train_dirs[0])
# print(x_test_dirs[0], y_test_dirs[0])


class FeatureDataset(Dataset):
    def __init__(self, x_dirs, y_dirs, part="real"):
        self.x_dirs = x_dirs
        self.y_dirs = y_dirs
        self.part = part
        self.x = None
        self.y = None
        self.file_paths_x = None
        self.file_paths_y = None

        # gtr files
        features_arr, file_paths_x = self._build_feature_data(self.x_dirs)
        self.x = features_arr
        self.file_paths_x = file_paths_x

        # ney files
        features_arr, file_paths_y = self._build_feature_data(self.y_dirs)
        self.y = features_arr
        self.file_paths_y = file_paths_y

    def _build_feature_data(self, dir_paths):
        features_arr = []
        paths_arr = []
        for dir_path in dir_paths:
            features, file_paths = self._get_features_in_dir(dir_path)
            features_arr.extend(features)
            paths_arr.extend(file_paths)

        features_arr = np.array(features_arr, dtype=np.float32)
        features_arr = np.expand_dims(features_arr, axis=1)
        paths_arr = np.array(paths_arr)
        return features_arr, paths_arr

    def _get_features_in_dir(self, dir_path):
        features = []
        file_paths = []
        sorted_files = sorted(list(Path(dir_path).iterdir()),
                              key=lambda x: int(x.stem.split("_")[1]))
        for f in sorted_files:
            with open(f, "rb") as handle:
                chunk = pickle.load(handle)
                if self.part == "real":
                    # extract real part and normalize
                    stft_data = (chunk["stft-data"][0] + 200.0) / 400.0
                else:
                    # extract imaginary part and normalize
                    stft_data = (chunk["stft-data"][1] + 200.0) / 400.0
                features.append(stft_data)
                file_paths.append(str(f))
        return features, file_paths

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.file_paths_x[index], self.file_paths_y[index]

    def __len__(self):
        return self.x.shape[0]


train_dataset = FeatureDataset(x_train_dirs, y_train_dirs, part="real")
test_dataset = FeatureDataset(x_test_dirs, y_test_dirs)

stft_train_data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=4,
    shuffle=True)

stft_test_data_loader = DataLoader(
    dataset=test_dataset,
    batch_size=4,
    shuffle=False)
