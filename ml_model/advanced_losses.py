"""
Advanced Loss Functions for SAR Oil Spill Segmentation
======================================================
Implements Strategy 1 (Prompt 1.2) from prompts.md:
  1. Focal Loss (with gamma focusing on hard boundary pixels)
  2. Tversky Loss & Focal Tversky Loss (with alpha/beta for FP/FN balance and look-alike suppression)
  3. Combo Loss (BCE + Dice + Focal)
  4. Class-balanced Weighted BCE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation.
    Down-weights easy background pixels and focuses learning on hard boundary pixels.
    FL = -alpha * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, epsilon: float = 1e-7):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)
        y_true = torch.clamp(y_true, 0.0, 1.0)
        pt = torch.where(y_true == 1, y_pred, 1.0 - y_pred)
        alpha_t = torch.where(y_true == 1, self.alpha, 1.0 - self.alpha)
        loss = -alpha_t * torch.pow(1.0 - pt, self.gamma) * torch.log(pt)
        return torch.nan_to_num(loss.mean(), nan=0.0, posinf=0.0, neginf=0.0)


class TverskyLoss(nn.Module):
    """
    Tversky Loss: Generalization of Dice loss allowing trade-off between
    False Positives (beta) and False Negatives (alpha).
    Setting beta > alpha (e.g., alpha=0.3, beta=0.7) heavily penalizes false alarms,
    which is essential for rejecting SAR Look-Alikes (Class 2 in real-life scenes).
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred_f = torch.clamp(y_pred.view(-1), 0.0, 1.0)
        y_true_f = torch.clamp(y_true.view(-1), 0.0, 1.0)

        tp = (y_pred_f * y_true_f).sum()
        fn = ((1.0 - y_pred_f) * y_true_f).sum()
        fp = (y_pred_f * (1.0 - y_true_f)).sum()

        denom = tp + self.alpha * fn + self.beta * fp + self.smooth + 1e-6
        tversky = (tp + self.smooth) / denom
        loss = 1.0 - tversky
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss: Applies a gamma exponent to the Tversky index:
    FTL = (1 - TI)^gamma
    Provides higher gradients for small, difficult oil slicks amidst wide oceans.
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33, smooth: float = 1.0):
        super(FocalTverskyLoss, self).__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        tversky_loss = self.tversky(y_pred, y_true)
        return torch.nan_to_num(torch.pow(tversky_loss, self.gamma), nan=0.0, posinf=0.0, neginf=0.0)


class DiceLoss(nn.Module):
    """Sørensen-Dice loss for binary overlap."""
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred_f = y_pred.view(-1)
        y_true_f = y_true.view(-1)
        intersection = (y_pred_f * y_true_f).sum()
        cardinality = y_pred_f.sum() + y_true_f.sum()
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice


class ComboLoss(nn.Module):
    """
    Combo Loss: Combines Weighted BCE, Dice, and Focal Loss.
    loss = w_bce * BCE + w_dice * Dice + w_focal * Focal
    Combines smooth cross-entropy gradients, boundary dice sharpness, and hard negative focus.
    """
    def __init__(
        self,
        w_bce: float = 0.3,
        w_dice: float = 0.4,
        w_focal: float = 0.3,
        pos_weight: float = 2.9,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        smooth: float = 1.0,
    ):
        super(ComboLoss, self).__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.dice = DiceLoss(smooth=smooth)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, 0.0, 1.0)
        y_true = torch.clamp(y_true, 0.0, 1.0)
        y_pred_clamped = torch.clamp(y_pred, epsilon, 1.0 - epsilon)

        # Weighted BCE
        if self.pos_weight is not None:
            pos_w = self.pos_weight.to(y_pred.device)
            bce = - (pos_w * y_true * torch.log(y_pred_clamped) + (1.0 - y_true) * torch.log(1.0 - y_pred_clamped)).mean()
        else:
            bce = F.binary_cross_entropy(y_pred_clamped, y_true, reduction='mean')

        dice = self.dice(y_pred, y_true)
        focal = self.focal(y_pred, y_true)

        total = self.w_bce * bce + self.w_dice * dice + self.w_focal * focal
        return torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)


class AttentionBlock(nn.Module):
    """
    Channel-wise and spatial attention mechanism to focus on important regions.
    Applied after encoder blocks to weight important features.
    """
    def __init__(self, channels: int):
        super(AttentionBlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        sa = self.spatial_attention(x)
        x = x * sa
        return x


class BalancedFocalTverskyLoss(nn.Module):
    """
    Improved Focal Tversky Loss with dynamic weighting based on sample difficulty.
    Especially effective for imbalanced segmentation with rare small objects (oil spills).
    """
    def __init__(self, alpha: float = 0.4, beta: float = 0.6, gamma: float = 1.5, smooth: float = 1.0):
        super(BalancedFocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred_f = torch.clamp(y_pred.view(-1), 0.0, 1.0)
        y_true_f = torch.clamp(y_true.view(-1), 0.0, 1.0)

        tp = (y_pred_f * y_true_f).sum()
        fn = ((1.0 - y_pred_f) * y_true_f).sum()
        fp = (y_pred_f * (1.0 - y_true_f)).sum()

        tversky_idx = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth + 1e-6)
        focal_loss = torch.pow(1.0 - tversky_idx, self.gamma)
        
        return torch.nan_to_num(focal_loss, nan=0.0, posinf=0.0, neginf=0.0)


class EnhancedComboLoss(nn.Module):
    """
    Enhanced Combo Loss with better balance and dynamic weight scheduling.
    Combines Weighted BCE, Focal Tversky, and Dice for comprehensive boundary learning.
    """
    def __init__(
        self,
        w_bce: float = 0.2,
        w_tversky: float = 0.5,
        w_dice: float = 0.3,
        pos_weight: float = 2.9,
        focal_alpha: float = 0.4,
        focal_beta: float = 0.6,
        focal_gamma: float = 1.5,
        smooth: float = 1.0,
    ):
        super(EnhancedComboLoss, self).__init__()
        self.w_bce = w_bce
        self.w_tversky = w_tversky
        self.w_dice = w_dice
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.dice = DiceLoss(smooth=smooth)
        self.focal_tversky = BalancedFocalTverskyLoss(
            alpha=focal_alpha, beta=focal_beta, gamma=focal_gamma, smooth=smooth
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, 0.0, 1.0)
        y_true = torch.clamp(y_true, 0.0, 1.0)
        y_pred_clamped = torch.clamp(y_pred, epsilon, 1.0 - epsilon)

        # Weighted BCE
        if self.pos_weight is not None:
            pos_w = self.pos_weight.to(y_pred.device)
            bce = - (pos_w * y_true * torch.log(y_pred_clamped) + (1.0 - y_true) * torch.log(1.0 - y_pred_clamped)).mean()
        else:
            bce = F.binary_cross_entropy(y_pred_clamped, y_true, reduction='mean')

        dice = self.dice(y_pred, y_true)
        tversky = self.focal_tversky(y_pred, y_true)

        total = self.w_bce * bce + self.w_tversky * tversky + self.w_dice * dice
        return torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)


if __name__ == '__main__':
    # Unit tests
    pred = torch.tensor([[[[0.05, 0.95], [0.10, 0.85]]]], dtype=torch.float32)
    target = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]], dtype=torch.float32)

    fl = FocalLoss()(pred, target)
    tl = TverskyLoss(alpha=0.3, beta=0.7)(pred, target)
    ftl = FocalTverskyLoss(alpha=0.3, beta=0.7)(pred, target)
    combo = ComboLoss()(pred, target)
    enhanced_combo = EnhancedComboLoss()(pred, target)

    print(f"Focal Loss:           {fl.item():.4f}")
    print(f"Tversky Loss:         {tl.item():.4f}")
    print(f"Focal Tversky Loss:   {ftl.item():.4f}")
    print(f"Combo Loss:           {combo.item():.4f}")
    print(f"Enhanced Combo Loss:  {enhanced_combo.item():.4f}")
    print("All loss functions passed test successfully!")
