import torch
import torch.nn as nn


class MagnitudeLoss(nn.Module):
    def __init__(self, mini, maxi):
        super().__init__()
        self.mini = mini
        self.maxi = maxi

    def forward(self, y_hat, y):
        y_hat = y_hat * (self.maxi - self.mini) + self.mini
        y = y * (self.maxi - self.mini) + self.mini
        magnitude_diff = y_hat - y
        loss = torch.mean(torch.abs(magnitude_diff))

        return loss


if __name__ == "__main__":
    predicted = torch.rand(4, 1, 256, 256)
    target = torch.rand(4, 1, 256, 256)
    criterion = MagnitudeLoss(1, 10)
    loss = criterion(predicted, target)
    print(loss)
