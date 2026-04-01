import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphFocalLoss(nn.Module):
    """
    Graph Focal Loss (Colab v82 Version).
    
    This version uses the internal pos_weight of BCE to precisely match the
    synergy detection gradients reported in the paper.
    """
    def __init__(self, gamma=4.0, pos_weight=1.5):
        super(GraphFocalLoss, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # Ensure targets are on the correct device and shape
        p = torch.sigmoid(logits)
        
        # Use the exact Colab implementation of weighted BCE
        pos_weight_tensor = torch.tensor([self.pos_weight], device=logits.device)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=pos_weight_tensor
        )
        
        # Calculate the probability of the true class
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Apply the focusing factor
        focal_loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        return focal_loss.mean()
