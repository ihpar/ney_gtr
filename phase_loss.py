import torch
import torch.nn.functional as F


def phase_loss(predicted_phase, target_phase):
    # Scale phase values from [0, 1] to [-pi, pi]
    predicted_phase = (predicted_phase - 0.5) * 2 * torch.pi
    target_phase = (target_phase - 0.5) * 2 * torch.pi

    # Compute phase difference and wrap it to the range [-pi, pi]
    phase_diff = predicted_phase - target_phase
    phase_diff = torch.atan2(torch.sin(phase_diff),
                             torch.cos(phase_diff))

    # Compute the loss (mean squared error on the wrapped phase difference)
    loss = torch.mean(phase_diff ** 2)

    return loss


if __name__ == "__main__":
    predicted = torch.rand(4, 1, 256, 256)
    target = torch.rand(4, 1, 256, 256)
    loss = phase_loss(predicted, target)
    print(loss)
    ang = torch.tensor([torch.pi])
    print(torch.atan2(torch.sin(ang), torch.cos(ang)))
