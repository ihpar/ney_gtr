import pickle
import numpy as np
from constants import *
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class FeatureDataset(Dataset):
    def __init__(self, x_dirs, y_dirs, representation, min_max, part):
        self.x_dirs = x_dirs
        self.y_dirs = y_dirs
        self.representation = representation
        self.min_max = min_max
        self.part = part

        # gtr files
        self.x, self.file_paths_x = self._build_feature_data(
            self.x_dirs,
            self.representation,
            self.min_max["gtr"],
            self.part
        )

        # ney files
        self.y, self.file_paths_y = self._build_feature_data(
            self.y_dirs,
            self.representation,
            self.min_max["ney"],
            self.part
        )

    def _build_feature_data(self, dir_paths, representation, min_max, part):
        features_arr = []
        paths_arr = []
        for dir_path in dir_paths:
            features, file_paths = self._get_features_in_dir(
                dir_path, representation,
                min_max, part
            )
            features_arr.extend(features)
            paths_arr.extend(file_paths)

        features_arr = np.array(features_arr, dtype=np.float32)

        paths_arr = np.array(paths_arr)
        return features_arr, paths_arr

    def _get_features_in_dir(self, dir_path, representation, min_max, part):
        features = []
        file_paths = []
        sorted_files = sorted(list(Path(dir_path).iterdir()),
                              key=lambda x: int(x.stem.split("_")[1]))
        for f in sorted_files:
            with open(f, "rb") as handle:
                chunk = pickle.load(handle)
            if representation == "polar":
                magni_data = chunk["abs"]
                magni_data = (magni_data - min_max["min"]["abs"]) /\
                    (min_max["max"]["abs"] - min_max["min"]["abs"])
                magni_data = np.log1p(magni_data)

                phase_data = chunk["angle"]
                phase_data = (phase_data - min_max["min"]["angle"]) /\
                    (min_max["max"]["angle"] - min_max["min"]["angle"])

                if part == "abs":
                    features.append(np.expand_dims(magni_data, axis=0))
                elif part == "ang":
                    features.append(np.expand_dims(phase_data, axis=0))
                else:
                    combined = np.array([magni_data, phase_data])
                    features.append(combined)
            elif representation == "cartesian":
                re_data = chunk["real"]
                re_data = (re_data - min_max["min"]["real"]) /\
                    (min_max["max"]["real"] - min_max["min"]["real"])
                re_data = np.log1p(re_data)

                im_data = chunk["imag"]
                im_data = (im_data - min_max["min"]["imag"]) /\
                    (min_max["max"]["imag"] - min_max["min"]["imag"])
                im_data = np.log1p(im_data)

                if part == "real":
                    features.append(np.expand_dims(re_data, axis=0))
                elif part == "imag":
                    features.append(np.expand_dims(im_data, axis=0))
                else:
                    combined = np.array([re_data, im_data])
                    features.append(combined)

            file_paths.append(str(f))
        return features, file_paths

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.file_paths_x[index], self.file_paths_y[index]

    def __len__(self):
        return self.x.shape[0]


def build_data_loaders(representation, min_max, part=None, test_size=0.2):
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
                                   representation, min_max, part)
    test_dataset = FeatureDataset(x_test_dirs, y_test_dirs,
                                  representation, min_max, part)

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True)

    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=4,
        shuffle=False)

    return train_data_loader, test_data_loader


if __name__ == "__main__":
    with open("dataset/features/min_max.pkl", "rb") as handle:
        min_max = pickle.load(handle)

    train_data_loader, _ = build_data_loaders("polar",
                                              min_max,
                                              part="abs")

    x, y, _, _ = next(iter(train_data_loader))
    print(x.size())
    print(y.size())
