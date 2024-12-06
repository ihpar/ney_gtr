import torch
import torch.nn as nn
import torch.nn.functional as F


class Model_1(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_features=64):
        super().__init__()
        # Encoder
        self.encoder1 = self._block(in_channels, base_features)
        self.encoder2 = self._block(base_features, base_features * 2)
        self.encoder3 = self._block(base_features * 2, base_features * 4)
        self.encoder4 = self._block(base_features * 4, base_features * 8)

        # Bottleneck
        self.bottleneck = self._block(base_features * 8, base_features * 16)

        # Decoder
        self.up4 = self._upsample(base_features * 16, base_features * 8)
        self.decoder4 = self._block(base_features * 16, base_features * 8)

        self.up3 = self._upsample(base_features * 8, base_features * 4)
        self.decoder3 = self._block(base_features * 8, base_features * 4)

        self.up2 = self._upsample(base_features * 4, base_features * 2)
        self.decoder2 = self._block(base_features * 4, base_features * 2)

        self.up1 = self._upsample(base_features * 2, base_features)
        self.decoder1 = self._block(base_features * 2, base_features)

        # Final output layer
        self.final_conv = nn.Conv2d(base_features, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Encoding path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoding path
        dec4 = self.decoder4(torch.cat([self.up4(bottleneck), enc4], dim=1))
        # dec4 = self.bn4(dec4)
        dec3 = self.decoder3(torch.cat([self.up3(dec4), enc3], dim=1))
        # dec3 = self.bn3(dec3)
        dec2 = self.decoder2(torch.cat([self.up2(dec3), enc2], dim=1))
        # dec2 = self.bn2(dec2)
        dec1 = self.decoder1(torch.cat([self.up1(dec2), enc1], dim=1))
        # dec1 = self.bn1(dec1)

        # Final output
        return self.activation(self.final_conv(dec1))

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def _upsample(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


if __name__ == "__main__":
    model = Model_1(1, 1, base_features=64)
    x = torch.randn(8, 1, 512, 512)
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
