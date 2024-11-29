import torch
import torch.nn as nn


class MagnitudeLoss(nn.Module):
    def __init__(self, mini, maxi):
        super().__init__()
        self.mini = mini
        self.maxi = maxi

    def forward(self, y_hat, y):
        y_hat = 0.5 * (y_hat + 1.0) * (self.maxi - self.mini) + self.mini
        y = 0.5 * (y + 1.0) * (self.maxi - self.mini) + self.mini
        magnitude_diff = y_hat - y
        loss = torch.mean(magnitude_diff ** 2)

        return loss


if __name__ == "__main__":
    predicted = torch.rand(4, 1, 256, 256)
    target = torch.rand(4, 1, 256, 256)
    criterion = MagnitudeLoss(1, 10)
    loss = criterion(predicted, target)
    print(loss)
