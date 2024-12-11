import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Causal convolution layer."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            # Remove extra padding for causality
            out = out[:, :, :-self.padding]
        return out


class ResidualBlock(nn.Module):
    """Residual block for WaveNet."""

    def __init__(self, residual_channels, skip_channels, kernel_size, dilation):
        super().__init__()
        self.filter_conv = CausalConv1d(
            residual_channels, residual_channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(
            residual_channels, residual_channels, kernel_size, dilation)
        self.residual_conv = nn.Conv1d(
            residual_channels, residual_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(
            residual_channels, skip_channels, kernel_size=1)

    def forward(self, x):
        filter_output = torch.tanh(self.filter_conv(x))
        gate_output = torch.sigmoid(self.gate_conv(x))
        output = filter_output * gate_output
        skip = self.skip_conv(output)
        residual = self.residual_conv(output)
        return x + residual, skip


class WaveNet(nn.Module):
    """WaveNet model."""

    def __init__(self, in_channels, residual_channels, skip_channels, n_blocks, n_layers, kernel_size=2):
        super().__init__()
        self.input_conv = CausalConv1d(
            in_channels, residual_channels, kernel_size=1, dilation=1)
        self.residual_blocks = nn.ModuleList()
        self.n_layers = n_layers

        for b in range(n_blocks):
            for l in range(n_layers):
                dilation = 2 ** l
                self.residual_blocks.append(
                    ResidualBlock(residual_channels,
                                  skip_channels, kernel_size, dilation)
                )

        self.final_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, in_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.input_conv(x)
        skip_connections = []

        for block in self.residual_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        output = sum(skip_connections)
        output = self.final_conv(output)
        return output


# Example usage and training loop
if __name__ == "__main__":
    batch_size = 4
    input_channels = 1
    residual_channels = 32
    skip_channels = 32
    n_blocks = 2
    n_layers = 10
    seq_length = 16000  # Example sequence length for audio

    model = WaveNet(
        in_channels=input_channels,
        residual_channels=residual_channels,
        skip_channels=skip_channels,
        n_blocks=n_blocks,
        n_layers=n_layers
    )

    example_input = torch.randn(batch_size, input_channels, seq_length)
    output = model(example_input)

    print("Input shape:", example_input.shape)
    print("Output shape:", output.shape)

    # Sample training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # # Example dataset and dataloader
    # class DummyDataset(torch.utils.data.Dataset):
    #     def __init__(self, size, seq_length):
    #         self.size = size
    #         self.seq_length = seq_length

    #     def __len__(self):
    #         return self.size

    #     def __getitem__(self, idx):
    #         x = torch.randn(1, self.seq_length)
    #         y = torch.randn(1, self.seq_length)
    #         return x, y

    # dataset = DummyDataset(size=4, seq_length=seq_length)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=batch_size, shuffle=True)

    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # num_epochs = 1
    # for epoch in range(num_epochs):
    #     model.train()
    #     epoch_loss = 0.0
    #     for batch_x, batch_y in dataloader:
    #         batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    #         optimizer.zero_grad()
    #         predictions = model(batch_x)
    #         loss = criterion(predictions, batch_y)
    #         loss.backward()
    #         optimizer.step()

    #         epoch_loss += loss.item()

    #     print(
    #         f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
