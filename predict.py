import torch
import librosa
import numpy as np
import torch.utils
import torch.utils.data
from constants import *
from utils import stitch_wave_chunks


def un_scale(data, mini, maxi, use_exp):
    if use_exp:
        data = np.expm1(data)
    return (data * (maxi - mini)) + mini


def predict(model: torch.nn.Module,
            data_loader,
            min_max,
            representation: str,
            limit=10,
            part: str = None
            ):

    model.eval()
    mm = min_max["ney"]
    predictions_0, predictions_1 = None, None
    targets_0, targets_1 = None, None
    keys = ["real", "imag"]
    if representation == "polar":
        keys = ["abs", "angle"]

    i = 0
    for x, y, x_paths, y_paths in data_loader:
        print(", ".join([p.replace("dataset/features/ac_gtr/", "")
              for p in x_paths]))
        print(", ".join([p.replace("dataset/features/ney/", "")
              for p in y_paths]))

        with torch.no_grad():
            predicted_chunks = model(x).numpy()

        y = y.numpy()

        # (4, 2, 256, 256)
        mini_0, maxi_0 = mm["min"][keys[0]], mm["max"][keys[0]]
        mini_1, maxi_1 = mm["min"][keys[1]], mm["max"][keys[1]]

        pred_0 = predicted_chunks[:, 0, :, :]
        pred_0 = un_scale(pred_0, mini_0, maxi_0, True)
        if part is None:
            pred_1 = predicted_chunks[:, 1, :, :]
            pred_1 = un_scale(pred_1, mini_1, maxi_1,
                              representation == "cartesian")

        target_0 = y[:, 0, :, :]
        target_0 = un_scale(target_0, mini_0, maxi_0, True)
        if part is None:
            target_1 = y[:, 1, :, :]
            target_1 = un_scale(target_1, mini_1, maxi_1,
                                representation == "cartesian")

        if predictions_0 is None:
            predictions_0 = np.copy(pred_0)
            targets_0 = np.copy(target_0)
            if part is None:
                predictions_1 = np.copy(pred_1)
                targets_1 = np.copy(target_1)
        else:
            predictions_0 = np.concatenate(
                (predictions_0, pred_0), axis=0)
            targets_0 = np.concatenate((targets_0, target_0), axis=0)
            if part is None:
                predictions_1 = np.concatenate(
                    (predictions_1, pred_1), axis=0)
                targets_1 = np.concatenate((targets_1, target_1), axis=0)

        print("-" * 50)
        i += x.size()[0]
        if i == limit:
            break

    return predictions_0, predictions_1, targets_0, targets_1


def get_phase(data_loader,
              instrument="ney",
              limit=10):
    i = 0
    angles = None
    for x, y, x_paths, y_paths in data_loader:
        if instrument == "gtr":
            angle = x.numpy().squeeze(axis=1)
            print(", ".join([p.replace("dataset/features/ac_gtr/", "")
                             for p in x_paths]))
        else:
            angle = y.numpy().squeeze(axis=1)
            print(", ".join([p.replace("dataset/features/ney/", "")
                             for p in y_paths]))
        angle = un_scale(angle, -np.pi, np.pi, False)

        if angles is None:
            angles = np.copy(angle)
        else:
            angles = np.concatenate((angles, angle),
                                    axis=0
                                    )

        print("-" * 50)
        i += x.size()[0]
        if i == limit:
            break
    return angles


def make_wav(chunks_0, chunks_1, representation):
    wave_chunks = []
    for chunk_0, chunk_1 in zip(chunks_0, chunks_1):
        if representation == "polar":
            chunk = chunk_0 * (np.cos(chunk_1) + 1j*np.sin(chunk_1))
        elif representation == "cartesian":
            chunk = chunk_0 + 1j * chunk_1
        wave_chunk = librosa.istft(chunk, n_fft=N_FFT, hop_length=HOP)
        wave_chunks.append(wave_chunk)

    stitched_wave = stitch_wave_chunks(wave_chunks)
    return stitched_wave


if __name__ == "__main__":
    import pickle
    from dataset import build_data_loaders

    representation = "polar"
    part = "ang"
    with open("dataset/features/min_max.pkl", "rb") as handle:
        min_max = pickle.load(handle)
    _, test_data_loader_phase = build_data_loaders(
        representation, min_max, part=part)
    phase = get_phase(test_data_loader_phase, instrument="ney", limit=12)
    print(phase.shape)
