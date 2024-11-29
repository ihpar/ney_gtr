from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import torch.optim as optim
import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super(UNetGenerator, self).__init__()
        self.encoder = nn.ModuleList([
            self._block(in_channels, features, kernel_size=4,
                        stride=2, padding=1),  # [128 -> 64]
            self._block(features, features * 2, 4, 2,
                        1),                          # [64 -> 32]
            self._block(features * 2, features * 4, 4, 2,
                        1),                      # [32 -> 16]
            self._block(features * 4, features * 8, 4, 2,
                        1),                      # [16 -> 8]
            self._block(features * 8, features * 8, 4, 2,
                        1),                      # [8 -> 4]
        ])

        self.decoder = nn.ModuleList([
            self._upblock(features * 8, features * 8, 4, 2,
                          1),                    # [4 -> 8]
            self._upblock(features * 8 * 2, features * 4, 4,
                          2, 1),                # [8 -> 16]
            self._upblock(features * 4 * 2, features * 2, 4,
                          2, 1),                # [16 -> 32]
            self._upblock(features * 2 * 2, features, 4, 2,
                          1),                    # [32 -> 64]
        ])

        self.final_layer = nn.ConvTranspose2d(
            features * 2, out_channels, kernel_size=4, stride=2, padding=1)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def _upblock(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        encodings = []

        # Encoding (downsampling)
        for layer in self.encoder:
            x = layer(x)
            encodings.append(x)

        # Decoding (upsampling)
        # Exclude last encoding and reverse for skip connections
        encodings = encodings[:-1][::-1]
        for i, layer in enumerate(self.decoder):
            x = layer(x)

            # Match size for concatenation
            if x.size(2) != encodings[i].size(2) or x.size(3) != encodings[i].size(3):
                encodings[i] = torch.nn.functional.interpolate(
                    encodings[i], size=(x.size(2), x.size(3)), mode='nearest')

            x = torch.cat([x, encodings[i]], dim=1)

        return torch.sigmoid(self.final_layer(x))


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=2, features=64):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            self._block(in_channels, features, 4, 2,
                        1),             # [128 -> 64]
            self._block(features, features * 2, 4, 2,
                        1),            # [64 -> 32]
            self._block(features * 2, features * 4,
                        4, 2, 1),        # [32 -> 16]
            self._block(features * 4, features * 8,
                        4, 1, 1),        # [16 -> 15]
            nn.Conv2d(features * 8, 1, kernel_size=4,
                      stride=1, padding=1)  # [15 -> 14]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))


# Models
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = UNetGenerator(in_channels=1, out_channels=1).to(device)
discriminator = PatchGANDiscriminator(in_channels=2).to(device)

# Hyperparameters
batch_size = 4
epochs = 1
lr = 2e-4
lambda_l1 = 100  # Weight for L1 loss

# Loss Functions
adversarial_loss = nn.BCELoss()  # Binary cross-entropy for adversarial loss
l1_loss = nn.L1Loss()  # L1 loss for reconstruction

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Fake and Real Labels
real_label = 1.0
fake_label = 0.0

# Training Loop


def train(dataloader):
    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        for batch_idx, (input_image, target_image) in enumerate(dataloader):
            input_image, target_image = input_image.to(
                device), target_image.to(device)

            # -------------------
            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()

            # Real pair
            real_pair = torch.cat((input_image, target_image), dim=1)
            real_output = discriminator(real_pair)
            real_loss = adversarial_loss(real_output, torch.ones_like(
                real_output, device=device) * real_label)

            # Fake pair
            fake_image = generator(input_image)
            fake_pair = torch.cat((input_image, fake_image), dim=1)
            # Detach to avoid updating G
            fake_output = discriminator(fake_pair.detach())
            fake_loss = adversarial_loss(fake_output, torch.zeros_like(
                fake_output, device=device) * fake_label)

            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # -------------------
            # Train Generator
            # -------------------
            optimizer_G.zero_grad()

            # Adversarial loss
            fake_output = discriminator(fake_pair)
            g_adv_loss = adversarial_loss(fake_output, torch.ones_like(
                fake_output, device=device) * real_label)

            # L1 loss for reconstruction
            g_l1_loss = l1_loss(fake_image, target_image) * lambda_l1

            # Total generator loss
            g_loss = g_adv_loss + g_l1_loss
            g_loss.backward()
            optimizer_G.step()

            # Log Progress
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(dataloader)} "
                      f"Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")

        # Save generated images for visualization
        save_image(fake_image, f"generated_epoch_{epoch}.png", normalize=True)

    print("Training Complete!")


# Example DataLoader (Dummy Data)
# Replace this with your dataset loader (e.g., torch.utils.data.DataLoader)


class DummyDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 1, 128, 128)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_image = self.data[idx]
        target_image = input_image * 2  # Dummy transformation
        return input_image, target_image


dataset = DummyDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train the Model
train(dataloader)
