import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W] (logits)
        # targets: [B, H, W] (labels)

        # Filter ignore_index
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        targets = targets[valid_mask]
        inputs = inputs.permute(0, 2, 3, 1)  # [B, H, W, C]
        inputs = inputs[valid_mask]  # [N, C]

        # Standard Cross Entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Focal term
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()