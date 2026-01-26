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


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=2):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W] (logits) -> Assuming C=1 for binary classification
        # targets: [B, H, W] (labels)
        
        # 1. Filter out ignore_index
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # 2. Apply Sigmoid activation to convert logits to probabilities (0-1)
        inputs = torch.sigmoid(inputs)

        # 3. Flatten tensors (only keeping valid pixels)
        # Ensure inputs and targets have the same shape for multiplication
        inputs = inputs.squeeze(1) # [B, H, W] (if C=1)
        
        inputs_flat = inputs[valid_mask]
        targets_flat = targets[valid_mask]

        # 4. Calculate Dice Coefficient
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)

        # 5. Return Loss (1 - Dice)
        return 1 - dice
    

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, weight_mask):
        # inputs: [B, C, H, W] (logits)
        # targets: [B, H, W] (labels)
        # weight_mask: [B, H, W] (weights)

        # Filter ignore_index
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # Apply the filter to targets and weight_mask
        targets = targets[valid_mask]
        
        # TODO maybe change weight_mask[y==1] = 10.0 so that deforestation pixels are weighted more??
        weight_mask = weight_mask[valid_mask]  # Filter weights to match valid pixels

        inputs = inputs.permute(0, 2, 3, 1)  # [B, H, W, C]
        inputs = inputs[valid_mask]  # [N, C]

        # Standard Cross Entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Apply the weight mask (Logic from src/models.py)
        ce_loss = ce_loss * weight_mask 
        
        pt = torch.exp(-ce_loss)

        # Focal term
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, inputs, targets, weight_mask):
        # inputs: logits, targets: 0/1, weight_mask: weights
        bce = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        # Apply mask
        weighted_bce = bce * weight_mask
        return weighted_bce.mean()

class WeightedDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true, weight_mask):
        # inputs: logits -> sigmoid
        y_pred = torch.sigmoid(y_pred)
        
        # Flatten
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        weight_mask = weight_mask.view(-1)
        
        # Weighted Intersection & Union
        intersection = (y_pred * y_true * weight_mask).sum()
        union = (y_pred * weight_mask).sum() + (y_true * weight_mask).sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = WeightedBCELoss()
        self.dice = WeightedDiceLoss()

    def forward(self, inputs, targets, weight_mask):
        return self.bce(inputs, targets, weight_mask) + self.dice(inputs, targets, weight_mask)