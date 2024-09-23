import torch
import torch.nn as nn


class Model_6(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 2, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(2, 4, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2),
            nn.ReLU()
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(1680, 8224),
            nn.Sigmoid(),
            nn.Linear(8224, 14940),
            nn.Sigmoid(),
            nn.Linear(14940, 29880),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 32, 3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, 3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, 3, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(4, -1)
        x = self.bottleneck(x)
        x = x.view(4, -1, 120, 249)
        x = self.decoder(x)

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

    # bottle
    x = x.view(4, -1)
    print("B1", x.size())
    x = nn.Linear(1680, 8224)(x)
    print("B2", x.size())
    x = nn.Linear(8224, 14940)(x)
    print("B3", x.size())
    x = nn.Linear(14940, 29880)(x)
    print("B4", x.size())
    x = x.view(4, -1, 120, 249)
    print("B5", x.size())
    print()

    # decoder
    print("Decoder:")
    x = nn.ConvTranspose2d(1, 32, 3, stride=1)(x)
    print("D1", x.size())
    x = nn.ConvTranspose2d(32, 16, 3, stride=1)(x)
    print("D2", x.size())
    x = nn.ConvTranspose2d(16, 4, 3, stride=1)(x)
    print("D3", x.size())
    x = nn.ConvTranspose2d(4, 1, 3, stride=1)(x)
    print("D4", x.size())


if __name__ == "__main__":
    x = torch.randn(4, 1, 128, 257)
    print("Original shape:", x.shape)
    print()
    test_sizes(x)
    print("Testing model:")
    m = Model_6()
    y = m(x)
    print(y.shape)
