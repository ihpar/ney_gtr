import torch
import torch.nn as nn
import torch.nn.functional as F

# Generator Module


class HiFiGANGenerator(nn.Module):
    def __init__(self, upsample_rates, kernel_sizes, resblock_kernel_sizes, resblock_dilations):
        super(HiFiGANGenerator, self).__init__()

        # Initial convolution
        self.initial_conv = nn.Conv1d(
            1, 128, kernel_size=7, stride=1, padding=3)

        # Upsampling layers
        self.upsampling_blocks = nn.ModuleList()
        in_channels = 128
        for upsample_rate, kernel_size in zip(upsample_rates, kernel_sizes):
            self.upsampling_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        in_channels,
                        in_channels // 2,
                        kernel_size=kernel_size,
                        stride=upsample_rate,
                        padding=(kernel_size - upsample_rate) // 2
                    ),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels //= 2

        # ResBlock layers
        self.resblocks = nn.ModuleList()
        for kernel_size, dilation_set in zip(resblock_kernel_sizes, resblock_dilations):
            self.resblocks.append(
                ResBlock(in_channels, kernel_size, dilation_set))

        # Final convolution
        self.final_conv = nn.Conv1d(
            in_channels, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self.initial_conv(x)
        for upsampling, resblock in zip(self.upsampling_blocks, self.resblocks):
            x = upsampling(x)
            x = resblock(x)
        x = torch.tanh(self.final_conv(x))
        return x


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size,
                          padding=d * (kernel_size - 1) // 2, dilation=d),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(channels, channels, kernel_size,
                          padding=d * (kernel_size - 1) // 2, dilation=d),
            ) for d in dilations
        ])

        # Projection layer to match residual dimensions if necessary
        self.projection = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        residual = x
        for layer in self.layers:
            print(x.size())
            out = layer(x)
            if residual.shape != out.shape:
                residual = self.projection(residual)
            x = residual + out
        return x

# Multi-Scale Discriminator


class HiFiGANDiscriminator(nn.Module):
    def __init__(self):
        super(HiFiGANDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            SubDiscriminator(),
            SubDiscriminator(),
            SubDiscriminator()
        ])

    def forward(self, x):
        outputs = []
        for i, discriminator in enumerate(self.discriminators):
            outputs.append(discriminator(x))
            if i < len(self.discriminators) - 1:
                x = F.avg_pool1d(x, kernel_size=4, stride=2, padding=1)
        return outputs


class SubDiscriminator(nn.Module):
    def __init__(self):
        super(SubDiscriminator, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),
            nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4),
            nn.Conv1d(64, 256, kernel_size=41,
                      stride=4, padding=20, groups=16),
            nn.Conv1d(256, 512, kernel_size=41,
                      stride=4, padding=20, groups=16),
            nn.Conv1d(512, 1024, kernel_size=41,
                      stride=4, padding=20, groups=16),
            nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2),
        ])
        self.output_layer = nn.Conv1d(
            1024, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.2)
        return self.output_layer(x)

# Loss Function


class HiFiGANLoss(nn.Module):
    def __init__(self):
        super(HiFiGANLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, real_outputs, fake_outputs):
        gen_loss = sum([self.mse_loss(fake, torch.ones_like(fake))
                       for fake in fake_outputs])
        disc_loss = sum([self.mse_loss(fake, torch.zeros_like(fake)) + self.mse_loss(real, torch.ones_like(real))
                         for real, fake in zip(real_outputs, fake_outputs)])
        return gen_loss, disc_loss


# Example Usage
if __name__ == "__main__":
    # Generator and Discriminator
    generator = HiFiGANGenerator(
        upsample_rates=[8, 8, 2, 2],
        kernel_sizes=[16, 16, 4, 4],
        resblock_kernel_sizes=[3, 3, 3],
        resblock_dilations=[[1, 3, 9], [1, 3, 9], [1, 3, 9]]
    )
    discriminator = HiFiGANDiscriminator()

    # Input
    batch_size = 4
    seq_length = 8192
    input_audio = torch.randn(batch_size, 1, seq_length)

    # Forward Pass
    fake_audio = generator(input_audio)
    disc_real = discriminator(input_audio)
    disc_fake = discriminator(fake_audio)

    # Loss
    loss_fn = HiFiGANLoss()
    gen_loss, disc_loss = loss_fn(disc_real, disc_fake)

    print("Generator Loss:", gen_loss.item())
    print("Discriminator Loss:", disc_loss.item())
