import torch
import torch.nn as nn
import torch.nn.functional as F


class DeeperUNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_features=64):
        super(DeeperUNet, self).__init__()
        # Encoder
        self.encoder1 = self._block(in_channels, base_features)
        self.encoder2 = self._block(base_features, base_features * 2)
        self.encoder3 = self._block(base_features * 2, base_features * 4)
        self.encoder4 = self._block(base_features * 4, base_features * 8)
        self.encoder5 = self._block(base_features * 8, base_features * 16)

        # Bottleneck
        self.bottleneck = self._block(base_features * 16, base_features * 32)

        # Decoder
        self.up5 = self._upsample(base_features * 32, base_features * 16)
        self.decoder5 = self._block(base_features * 32, base_features * 16)

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

    def forward(self, x):
        # Encoding path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        enc5 = self.encoder5(F.max_pool2d(enc4, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc5, 2))

        # Decoding path
        dec5 = self.decoder5(torch.cat([self.up5(bottleneck), enc5], dim=1))
        dec4 = self.decoder4(torch.cat([self.up4(dec5), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([self.up3(dec4), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([self.up1(dec2), enc1], dim=1))

        # Final output
        return self.final_conv(dec1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def _upsample(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


def adversarial_loss(pred, target_is_real):
    """
    Adversarial loss using BCEWithLogitsLoss.
    pred: logits from the discriminator.
    target_is_real: True if the target is real, False if fake.
    """
    target = torch.ones_like(
        pred) if target_is_real else torch.zeros_like(pred)
    return nn.BCEWithLogitsLoss()(pred, target)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels, base_features=64):
        super(PatchDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_features,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_features, base_features * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_features * 2, base_features * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_features * 4, base_features * 8,
                      kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(base_features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_features * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)  # Outputs raw logits


if __name__ == "__main__":
    input = torch.rand((4, 1, 128, 128))
    model = DeeperUNet(1, 1)
    output = model(input)
    print(output.size())

    # Initialize models
    # Modify for your task
    generator = DeeperUNet(in_channels=1, out_channels=1)
    discriminator = PatchDiscriminator(in_channels=1)

    # Optimizers
    lr = 0.0002
    betas = (0.5, 0.999)
    gen_optimizer = torch.optim.Adam(
        generator.parameters(), lr=lr, betas=betas)
    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=lr, betas=betas)

    # Loss functions
    l1_loss = nn.L1Loss()  # For pixel-wise loss
    adv_loss = adversarial_loss  # Adversarial loss

    # Training loop
    for epoch in range(1):
        for _ in [1]:
            real = torch.rand((4, 1, 128, 128))  # Real spectrograms
            input = torch.rand((4, 1, 128, 128))  # Input spectrograms

            # ----------------------
            # Train Discriminator
            # ----------------------
            fake = generator(input)  # Generate fake spectrograms
            disc_optimizer.zero_grad()

            # Real loss
            pred_real = discriminator(real)
            loss_real = adv_loss(pred_real, target_is_real=True)

            # Fake loss
            # Detach to avoid affecting generator
            pred_fake = discriminator(fake.detach())
            loss_fake = adv_loss(pred_fake, target_is_real=False)

            # Total discriminator loss
            disc_loss = (loss_real + loss_fake) * 0.5
            disc_loss.backward()
            disc_optimizer.step()

            # ----------------------
            # Train Generator
            # ----------------------
            gen_optimizer.zero_grad()

            # Adversarial loss (Generator wants the discriminator to classify fake as real)
            pred_fake = discriminator(fake)
            gen_adv_loss = adv_loss(pred_fake, target_is_real=True)

            # Pixel-wise loss
            gen_pixel_loss = l1_loss(fake, real)

            # Total generator loss
            gen_loss = gen_adv_loss + 100 * gen_pixel_loss  # Weight pixel loss higher
            gen_loss.backward()
            gen_optimizer.step()

            # ----------------------
            # Logging
            # ----------------------
            print(
                f"Epoch [{epoch}/{1}] "
                f"Discriminator Loss: {disc_loss.item():.4f}, "
                f"Generator Loss: {gen_loss.item():.4f}"
            )
