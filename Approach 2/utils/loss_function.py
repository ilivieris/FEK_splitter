import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Focal loss function
    """
    def __init__(self, gamma=3., reduction='mean'):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)

        pt = torch.exp(-CE_loss)
        F_loss = ((1 - pt)**self.gamma) * CE_loss
        
        if self.reduction == 'sum':
            return F_loss.sum()
        elif self.reduction == 'mean':
            return F_loss.mean()