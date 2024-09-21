import torch
import torch.nn as nn


class Model_4(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(32896, 32896*2),
            nn.Sigmoid(),
            nn.Linear(32896*2, 32896),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(4, -1)
        x = self.fc(x)
        x = x.view(4, 1, 128, 257)

        return x


def test_sizes(x):
    x = x.view(4, -1)
    print("L1", x.size())
    x = nn.Linear(32896, 32896*2)(x)
    print("L2", x.size())
    x = nn.Linear(32896*2, 32896)(x)
    print("L3", x.size())
    x = x.view(4, 1, 128, 257)
    print("L4", x.size())


if __name__ == "__main__":
    x = torch.randn(4, 1, 128, 257)
    print("Original shape:", x.shape)
    print()
    test_sizes(x)
    print("Testing model:")
    m = Model_4()
    y = m(x)
    print(y.shape)
