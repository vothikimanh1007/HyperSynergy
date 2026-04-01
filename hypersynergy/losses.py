import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphFocalLoss(nn.Module):
    """
    Calibrated Graph Focal Loss for HyperSynergy.
    
    Specifically optimized for the DoTatLoi-714 benchmark's 1:5 class imbalance 
    (95.5% data sparsity). It prevents the overwhelming majority of negative 
    samples (non-synergistic herb pairs) from washing out the gradient during 
    the MATG optimization.
    
    Args:
        alpha (float): Weighting factor for the positive class (default: 1.5).
        gamma (float): Focusing parameter to down-weight easy examples (default: 4.0).
        reduction (str): Specifies the reduction to apply to the output ('none', 'mean', 'sum').
    """
    def __init__(self, alpha=1.5, gamma=4.0, reduction='mean'):
        super(GraphFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Calculates the focal loss.
        
        Args:
            inputs (Tensor): The raw logits from the MATG decoder (before sigmoid).
            targets (Tensor): Ground truth binary labels (0.0 or 1.0).
            
        Returns:
            Tensor: The computed focal loss.
        """
        # Calculate standard Binary Cross Entropy Loss (with logits for numerical stability)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate pt (the model's estimated probability for the correct class)
        pt = torch.exp(-bce_loss)
        
        # Apply the alpha balancing weight (only scale positive targets by alpha)
        alpha_t = targets * self.alpha + (1 - targets) * 1.0
        
        # Compute the final focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
