import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.residual = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        # Standard convolutional path
        residual = self.residual(x)  # Adjust the channel dimensions if needed
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        # Residual connection
        out += residual
        return self.relu(out)


class UNetResidual(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder (Downsampling path)
        self.enc1 = ResidualBlock(in_channels, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)
        self.enc4 = ResidualBlock(256, 512)

        # Bottleneck
        self.bottleneck = ResidualBlock(512, 1024)

        # Decoder (Upsampling path)
        self.dec4 = self.upconv_block(1024, 512)
        self.dec3 = self.upconv_block(512, 256)
        self.dec2 = self.upconv_block(256, 128)
        self.dec1 = self.upconv_block(128, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def upconv_block(self, in_channels, out_channels):
        """Upsampling block using ConvTranspose2d"""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # [4, 64, 128, 128]
        enc2 = self.enc2(self.downsample(enc1))  # [4, 128, 64, 64]
        enc3 = self.enc3(self.downsample(enc2))  # [4, 256, 32, 32]
        enc4 = self.enc4(self.downsample(enc3))  # [4, 512, 16, 16]

        # Bottleneck
        bottleneck = self.bottleneck(self.downsample(enc4))  # [4, 1024, 8, 8]

        # Decoder with skip and residual connections
        dec4 = self.dec4(bottleneck)  # [4, 512, 16, 16]
        # Align sizes before concatenation
        dec4 = self._crop_and_concat(enc4, dec4)
        dec4 = ResidualBlock(1024, 512)(dec4)

        dec3 = self.dec3(dec4)  # [4, 256, 32, 32]
        dec3 = self._crop_and_concat(enc3, dec3)
        dec3 = ResidualBlock(512, 256)(dec3)

        dec2 = self.dec2(dec3)  # [4, 128, 64, 64]
        dec2 = self._crop_and_concat(enc2, dec2)
        dec2 = ResidualBlock(256, 128)(dec2)

        dec1 = self.dec1(dec2)  # [4, 64, 128, 128]
        dec1 = self._crop_and_concat(enc1, dec1)
        dec1 = ResidualBlock(128, 64)(dec1)

        # Final output layer
        return self.final_conv(dec1)

    def downsample(self, x):
        """Downsample by max pooling"""
        return nn.functional.max_pool2d(x, kernel_size=2)

    def _crop_and_concat(self, enc_feature, dec_feature):
        """Manually crop the encoder feature map to match the decoder feature map size."""
        _, _, h_dec, w_dec = dec_feature.size()
        _, _, h_enc, w_enc = enc_feature.size()

        # Calculate cropping margins
        crop_h = (h_enc - h_dec) // 2
        crop_w = (w_enc - w_dec) // 2

        # Crop the encoder feature map
        enc_feature_cropped = enc_feature[
            :, :, crop_h: crop_h + h_dec, crop_w: crop_w + w_dec
        ]

        # Concatenate along the channel dimension
        return torch.cat([enc_feature_cropped, dec_feature], dim=1)


if __name__ == "__main__":
    # Instantiate the model
    model = UNetResidual(in_channels=1, out_channels=1)

    # Test with a dummy input
    x = torch.randn(4, 1, 128, 128)  # Batch of 4, 1-channel input, 128x128
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be [4, 1, 128, 128]
