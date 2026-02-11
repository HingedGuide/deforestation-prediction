import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, inputs, targets, weight_mask):
        # 1. Zorg voor gelijke shapes
        # inputs: [B, 1, H, W] -> [B, H, W]
        if inputs.ndim > targets.ndim:
            inputs = inputs.squeeze(1)
        
        # 2. Bereken loss (zonder reductie, dus per pixel)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        
        # 3. Pas weights toe
        weighted_loss = bce_loss * weight_mask
        
        # 4. Return gemiddelde
        return weighted_loss.mean()

class WeightedDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets, weight_mask):
        # 1. Zorg voor gelijke shapes
        if inputs.ndim > targets.ndim:
            inputs = inputs.squeeze(1)
            
        # 2. Logits -> Probabilities
        probs = torch.sigmoid(inputs)
        
        # 3. Flatten (zodat we dot products kunnen doen)
        probs = probs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        weight_mask = weight_mask.contiguous().view(-1)
        
        # 4. Weighted Intersection en Union
        intersection = (probs * targets * weight_mask).sum()
        union = (probs * weight_mask).sum() + (targets * weight_mask).sum()
        
        # 5. Dice Score
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = WeightedBCELoss()
        self.dice = WeightedDiceLoss()

    def forward(self, inputs, targets, weight_mask):
        return self.bce(inputs, targets, weight_mask) + self.dice(inputs, targets, weight_mask)