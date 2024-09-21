import pickle
import librosa
import numpy as np
from constants import *
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class FeatureDataset(Dataset):
    def __init__(self, x_dirs, y_dirs, part):
        self.x_dirs = x_dirs
        self.y_dirs = y_dirs
        self.part = part
        self.x = None
        self.y = None
        self.file_paths_x = None
        self.file_paths_y = None
        self.min_max_vals = {
            "gtr": {"min": None, "max": None},
            "ney": {"min": None, "max": None}
        }

        # gtr files
        features_arr, file_paths_x, max_gtr, min_gtr = self._build_feature_data(
            self.x_dirs)
        self.x = features_arr
        self.file_paths_x = file_paths_x
        self.min_max_vals["gtr"]["max"] = max_gtr
        self.min_max_vals["gtr"]["min"] = min_gtr

        # ney files
        features_arr, file_paths_y, max_ney, min_ney = self._build_feature_data(
            self.y_dirs)
        self.y = features_arr
        self.file_paths_y = file_paths_y
        self.min_max_vals["ney"]["max"] = max_ney
        self.min_max_vals["ney"]["min"] = min_ney

    def _build_feature_data(self, dir_paths):
        features_arr = []
        paths_arr = []
        for dir_path in dir_paths:
            features, file_paths = self._get_features_in_dir(dir_path)
            features_arr.extend(features)
            paths_arr.extend(file_paths)

        features_arr = np.array(features_arr, dtype=np.float32)
        min_val = np.min(features_arr)
        max_val = np.max(features_arr)
        features_arr = (features_arr - min_val) / (max_val - min_val)
        features_arr = np.expand_dims(features_arr, axis=1)
        paths_arr = np.array(paths_arr)
        return features_arr, paths_arr, max_val, min_val

    def _get_features_in_dir(self, dir_path):
        features = []
        file_paths = []
        sorted_files = sorted(list(Path(dir_path).iterdir()),
                              key=lambda x: int(x.stem.split("_")[1]))
        for f in sorted_files:
            with open(f, "rb") as handle:
                chunk = pickle.load(handle)
                if self.part == "abs":
                    # extract abs and convert to db
                    stft_data = librosa.power_to_db(chunk["abs"])
                elif self.part == "angle":
                    # extract angle
                    stft_data = chunk["angle"]
                else:
                    raise Exception(
                        "part parameter must be either `abs` or `angle`")
                features.append(stft_data)
                file_paths.append(str(f))
        return features, file_paths

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.file_paths_x[index], self.file_paths_y[index]

    def __len__(self):
        return self.x.shape[0]


def build_data_loaders(part):
    ney_feature_dirs = sorted(
        [f for f in Path(NEY_FEATURE_DIR).iterdir() if f.is_dir()])
    gtr_feature_dirs = sorted(
        [f for f in Path(GTR_FEATURE_DIR).iterdir() if f.is_dir()])

    x_train_dirs, x_test_dirs, y_train_dirs, y_test_dirs = train_test_split(
        gtr_feature_dirs, ney_feature_dirs, test_size=0.2, random_state=42)

    train_dataset = FeatureDataset(x_train_dirs, y_train_dirs, part)
    test_dataset = FeatureDataset(x_test_dirs, y_test_dirs, part)

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True)

    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=4,
        shuffle=False)

    return train_data_loader, test_data_loader
