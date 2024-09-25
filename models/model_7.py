import torch
import torch.nn as nn


class Model_7(nn.Module):
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
            nn.Sigmoid()
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(1680, 1680),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, (3, 4), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, (4, 3), stride=2),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(32896, 32896),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(4, -1)
        x = self.bottleneck(x)
        x = x.view(4, 16, 7, -1)
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

    # bottle
    x = x.view(4, -1)
    print("B1", x.size())
    x = nn.Linear(1680, 1680)(x)
    print("B2", x.size())
    x = x.view(4, 16, 7, -1)
    print("B3", x.size())
    print()

    # decoder
    print("Decoder:")
    x = nn.ConvTranspose2d(16, 8, 3, stride=2)(x)
    print("D1", x.size())
    x = nn.ConvTranspose2d(8, 4, 3, stride=2)(x)
    print("D2", x.size())
    x = nn.ConvTranspose2d(4, 2, (3, 4), stride=2)(x)
    print("D3", x.size())
    x = nn.ConvTranspose2d(2, 1, (4, 3), stride=2)(x)
    print("D4", x.size())


if __name__ == "__main__":
    print()
    x = torch.randn(4, 1, 128, 257)
    print("Original shape:")
    print(x.shape)
    print()
    test_sizes(x)
    print("Testing model:")
    m = Model_7()
    y = m(x)
    print(y.shape)
