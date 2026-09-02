import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Dice = 2 * (pred * target).sum() / (pred.sum() + target.sum() + smooth)
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Flatten predictions and targets
        y_pred_f = y_pred.view(-1)
        y_true_f = y_true.view(-1)
        
        intersection = (y_pred_f * y_true_f).sum()
        union = y_pred_f.sum() + y_true_f.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice

class BCEDiceLoss(nn.Module):
    """
    Hybrid Loss: combines Binary Cross Entropy Loss and Dice Loss.
    We apply pos_weight in BCE to handle the class imbalance (bg/oil = 2.9x).
    """
    def __init__(self, alpha=0.5, pos_weight=2.9, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.dice = DiceLoss(smooth=smooth)
        
        # Binary Cross Entropy with raw logits is generally more numerically stable,
        # but since our model output has a Sigmoid built-in, we will use BCELoss on probability maps.
        # Alternatively, we could use BCEWithLogitsLoss by passing raw logits, but keeping BCELoss is simple.
        # To avoid log(0) issues, we clip predictions inside BCELoss or use torch.clamp.
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight is not None else None

    def forward(self, y_pred, y_true):
        # Clip y_pred to prevent log(0) errors
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
        
        # Weighted Binary Cross Entropy:
        # loss = - (pos_weight * y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
        if self.pos_weight is not None:
            pos_w = self.pos_weight.to(y_pred.device)
            bce = - (pos_w * y_true * torch.log(y_pred) + (1.0 - y_true) * torch.log(1.0 - y_pred)).mean()
        else:
            bce = nn.functional.binary_cross_entropy(y_pred, y_true, reduction='mean')
            
        dice = self.dice(y_pred, y_true)
        
        return self.alpha * bce + (1.0 - self.alpha) * dice

if __name__ == '__main__':
    # Sanity check loss functions
    loss_fn = BCEDiceLoss(alpha=0.5, pos_weight=2.9)
    y_pred = torch.tensor([[[[0.1, 0.9], [0.2, 0.8]]]]) # (1, 1, 2, 2)
    y_true = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]]) # (1, 1, 2, 2)
    loss = loss_fn(y_pred, y_true)
    print("Computed hybrid loss value:", loss.item())
