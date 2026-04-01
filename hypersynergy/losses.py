import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphFocalLoss(nn.Module):
    """
    Graph Focal Loss (Colab v82 Version).
    
    Revised to align with the ModelEvaluator signature: alpha=1.5, gamma=4.0.
    This ensures that the weighting (alpha) and the focusing factor (gamma)
    are correctly applied to resolve the 1:5 class imbalance.
    """
    def __init__(self, alpha=1.5, gamma=4.0):
        super(GraphFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # Ensure targets are on the correct device and shape
        p = torch.sigmoid(logits)
        
        # Use alpha as the positive weight for the imbalanced minority class
        alpha_tensor = torch.tensor([self.alpha], device=logits.device)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=alpha_tensor
        )
        
        # Calculate the probability of the true class (p_t)
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Apply the focusing factor (1 - p_t)^gamma
        focal_loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        return focal_loss.mean()
