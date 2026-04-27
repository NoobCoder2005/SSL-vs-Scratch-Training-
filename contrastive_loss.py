import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        batch_size = z1.size(0)

        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate
        z = torch.cat([z1, z2], dim=0)  # (2N, D)

        # Similarity matrix
        sim = torch.matmul(z, z.T)  # (2N, 2N)

        # Remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim = sim.masked_fill(mask, -1e9)

        # Scale
        sim = sim / self.temperature

        # Positive pairs
        pos = torch.cat([
            torch.diag(sim, batch_size),
            torch.diag(sim, -batch_size)
        ], dim=0)

        # Denominator (logsumexp over rows)
        loss = -pos + torch.logsumexp(sim, dim=1)

        return loss.mean()