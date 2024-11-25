import torch
import torch.nn as nn
from phase_loss import phase_loss


def train_model(model: nn.Module,
                criterion,
                optimizer,
                device,
                train_data_loader,
                test_data_loader,
                early_stop,
                num_epochs=1,
                phase_loss_fun=None,
                minimaxi=None):

    model.to(device)
    history = {"train": [], "val": []}
    if minimaxi is not None:
        mini = minimaxi[0]
        maxi = minimaxi[1]

    print("Starting training...")

    for epoch in range(num_epochs):
        model.train(True)
        running_loss = 0.0
        num_train_batches = 0
        for gtr_features, ney_features, _, _ in train_data_loader:
            gtr_features = gtr_features.to(device)
            ney_features = ney_features.to(device)

            y_hat = model(gtr_features)
            if phase_loss_fun is not None:
                y_hat_mag = y_hat[:, 0:1, :, :] * 2.0 * torch.pi
                y_mag = ney_features[:, 0:1, :, :] * 2.0 * torch.pi
                mag_loss = criterion(y_hat_mag, y_mag)

                y_hat_phase = y_hat[:, 1:2, :, :]
                y_phase = ney_features[:, 1:2, :, :]
                phase_loss = phase_loss_fun(y_hat_phase, y_phase)

                loss = mag_loss + phase_loss
            else:
                if minimaxi is not None:
                    loss = criterion(torch.expm1(y_hat) * (maxi - mini) + mini,
                                     torch.expm1(ney_features) * (maxi - mini) + mini)
                else:
                    loss = criterion(y_hat, ney_features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_train_batches += 1

        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for gtr_features, ney_features, _, _ in test_data_loader:
                if len(gtr_features) != 4:
                    continue

                gtr_features = gtr_features.to(device)
                ney_features = ney_features.to(device)

                y_hat = model(gtr_features)
                if phase_loss_fun is not None:
                    y_hat_mag = y_hat[:, 0:1, :, :] * 2.0 * torch.pi
                    y_mag = ney_features[:, 0:1, :, :] * 2.0 * torch.pi
                    mag_loss = criterion(y_hat_mag, y_mag)

                    y_hat_phase = y_hat[:, 1:2, :, :]
                    y_phase = ney_features[:, 1:2, :, :]
                    phase_loss = phase_loss_fun(y_hat_phase, y_phase)

                    loss = mag_loss + phase_loss
                else:
                    if minimaxi is not None:
                        loss = criterion(torch.expm1(y_hat) * (maxi - mini) + mini,
                                         torch.expm1(ney_features) * (maxi - mini) + mini)
                    else:
                        loss = criterion(y_hat, ney_features)

                val_loss += loss.item()
                num_val_batches += 1

        train_loss = running_loss / num_train_batches
        val_loss = val_loss / num_val_batches

        print(
            f"E: {epoch + 1:03d}/{num_epochs}\t T: {train_loss:.6f}\t V: {val_loss:.6f}")

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if early_stop.early_stop(val_loss):
            print("Stopping early...")
            break

    return model, history


if __name__ == "__main__":
    import pickle
    import torch.optim as optim
    from models.model_17 import Model_17
    from dataset import build_data_loaders
    from early_stopper import EarlyStopper
    with open("dataset/features/min_max.pkl", "rb") as handle:
        min_max = pickle.load(handle)

    device = torch.device("cpu")
    representation = "polar"
    model = Model_17()
    train_data_loader, test_data_loader = build_data_loaders(
        representation, min_max, part=None)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    es = EarlyStopper(5, 3e-6)
    model, history = train_model(
        model,
        criterion,
        optimizer,
        device,
        train_data_loader, test_data_loader,
        es,
        num_epochs=1,
        phase_loss_fun=phase_loss
    )
