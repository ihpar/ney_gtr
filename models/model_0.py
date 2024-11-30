import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """A convolutional block with Conv2D -> BatchNorm -> ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Model_0(nn.Module):
    def __init__(self, in_channels=1, num_features=64):
        super().__init__()

        # Encoder blocks
        self.enc1 = ConvBlock(in_channels, num_features)
        self.enc2 = ConvBlock(num_features, num_features * 2)
        self.enc3 = ConvBlock(num_features * 2, num_features * 4)
        self.enc4 = ConvBlock(num_features * 4, num_features * 8)

        # Bottleneck
        self.bottleneck = ConvBlock(num_features * 8, num_features * 16)

        # Decoder blocks
        self.dec4 = ConvBlock(num_features * 16, num_features * 8)
        self.dec3 = ConvBlock(num_features * 8, num_features * 4)
        self.dec2 = ConvBlock(num_features * 4, num_features * 2)
        self.dec1 = ConvBlock(num_features * 2, num_features)

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2, 2)
        self.up4 = nn.ConvTranspose2d(
            num_features * 16, num_features * 8, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(
            num_features * 8, num_features * 4, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(
            num_features * 4, num_features * 2, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(
            num_features * 2, num_features, kernel_size=2, stride=2)

        # Final convolution
        self.final = nn.Conv2d(num_features, 1, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder path
        dec4 = self.dec4(torch.cat([self.up4(bottleneck), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))

        # Final output
        return self.tanh(self.final(dec1))


if __name__ == "__main__":
    model = Model_0(in_channels=1, num_features=32)
    x = torch.randn(4, 1, 256, 256)
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
