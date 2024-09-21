import torch
import torch.nn as nn


class Model_3(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 2, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(2, 4, 3, stride=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, 3, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(8385, 16448),
            nn.Sigmoid(),
            nn.Linear(16448, 32896),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(4, -1)
        x = self.fc(x)
        x = x.view(4, 1, 128, 257)

        return x


def test_sizes(x):
    # encoder
    print("Encoder:")
    x = nn.Conv2d(1, 2, 3, stride=2)(x)
    print("E1", x.size())
    x = nn.Conv2d(2, 4, 3, stride=2)(x)
    print("E2", x.size())
    x = nn.Conv2d(4, 8, 3, stride=2)(x)
    print("E4", x.size())
    x = nn.Conv2d(8, 16, 3, stride=2)(x)
    print("E5", x.size())
    print()

    # decoder
    print("Decoder:")
    x = nn.ConvTranspose2d(16, 8, 3, stride=2)(x)
    print("D1", x.size())
    x = nn.ConvTranspose2d(8, 4, 3, stride=2)(x)
    print("D2", x.size())
    x = nn.ConvTranspose2d(4, 2, 3, stride=2)(x)
    print("D3", x.size())
    x = nn.ConvTranspose2d(2, 1, 3, stride=1)(x)
    print("D4", x.size())
    x = x.view(4, -1)
    print("D5", x.size())
    x = nn.Linear(8385, 16448)(x)
    print("D6", x.size())
    x = nn.Linear(16448, 32896)(x)
    print("D6", x.size())
    x = x.view(4, 1, 128, 257)
    print("D7", x.size())


if __name__ == "__main__":
    x = torch.randn(4, 1, 128, 257)
    print("Original shape:", x.shape)
    print()
    test_sizes(x)
    print("Testing model:")
    m = Model_3()
    y = m(x)
    print(y.shape)
