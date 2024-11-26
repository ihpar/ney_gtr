import pickle
import numpy as np
from constants import *
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class FeatureDataset(Dataset):
    def __init__(self, x_dirs, y_dirs, min_max, part, log_scale):
        self.x_dirs = x_dirs
        self.y_dirs = y_dirs
        self.min_max = min_max
        self.part = part
        self.log_scale = log_scale

        # gtr files
        self.x, self.file_paths_x = self._build_feature_data(
            self.x_dirs,
            self.min_max["gtr"],
            self.part
        )

        # ney files
        self.y, self.file_paths_y = self._build_feature_data(
            self.y_dirs,
            self.min_max["ney"],
            self.part
        )

    def _build_feature_data(self, dir_paths, min_max, part):
        features_arr = []
        paths_arr = []
        for dir_path in dir_paths:
            features, file_paths = self._get_features_in_dir(
                dir_path,
                min_max,
                part
            )
            features_arr.extend(features)
            paths_arr.extend(file_paths)

        features_arr = np.array(features_arr, dtype=np.float32)

        paths_arr = np.array(paths_arr)
        return features_arr, paths_arr

    def _get_features_in_dir(self, dir_path, min_max, part):
        features = []
        file_paths = []
        sorted_files = sorted(list(Path(dir_path).iterdir()),
                              key=lambda x: int(x.stem.split("_")[1]))
        for f in sorted_files:
            with open(f, "rb") as handle:
                chunk = pickle.load(handle)

            mags = chunk["magnitude"]
            min_mag = min_max["min"]["magnitude"]
            max_mag = min_max["max"]["magnitude"]
            mags = (mags - min_mag) / (max_mag - min_mag)
            if self.log_scale:
                mags = np.log1p(mags)

            phases = chunk["phase"]
            min_phase = min_max["min"]["phase"]
            max_phase = min_max["max"]["phase"]
            phases = (phases - min_phase) / (max_phase - min_phase)

            dbs = chunk["db"]
            min_db = min_max["min"]["db"]
            max_db = min_max["max"]["db"]
            dbs = (dbs - min_db) / (max_db - min_db)
            if self.log_scale:
                dbs = np.log1p(dbs)

            if part == "magnitude":
                features.append(np.expand_dims(mags, axis=0))
            elif part == "phase":
                features.append(np.expand_dims(phases, axis=0))
            elif part == "db":
                features.append(np.expand_dims(dbs, axis=0))
            else:
                combined = np.array([mags, phases])
                features.append(combined)

            file_paths.append(str(f))
        return features, file_paths

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.file_paths_x[index], self.file_paths_y[index]

    def __len__(self):
        return self.x.shape[0]


def build_data_loaders(min_max, part=None, log_scale=False, test_size=0.2):
    gtr_feature_dirs = sorted(
        [f for f in Path(GTR_FEATURE_DIR).iterdir() if f.is_dir()])
    ney_feature_dirs = sorted(
        [f for f in Path(NEY_FEATURE_DIR).iterdir() if f.is_dir()])

    x_train_dirs, x_test_dirs, y_train_dirs, y_test_dirs = train_test_split(
        gtr_feature_dirs,
        ney_feature_dirs,
        test_size=test_size,
        random_state=42)

    train_dataset = FeatureDataset(x_train_dirs, y_train_dirs,
                                   min_max, part, log_scale)
    test_dataset = FeatureDataset(x_test_dirs, y_test_dirs,
                                  min_max, part, log_scale)

    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=4,
                                   shuffle=True)

    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=4,
                                  shuffle=False)

    return train_data_loader, test_data_loader


if __name__ == "__main__":
    with open("dataset/features/min_max.pkl", "rb") as handle:
        min_max = pickle.load(handle)

    _, test_loader = build_data_loaders(min_max,
                                        part="db",
                                        log_scale=False,
                                        test_size=0.15)

    x, y, _, _ = next(iter(test_loader))
    print(x.size())
    print(y.size())

    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(y[0].numpy().squeeze(axis=0))
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
