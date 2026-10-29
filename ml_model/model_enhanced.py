"""
Enhanced U-Net Architectures for SAR Oil Spill Detection
========================================================
Implements Strategy 2 (Prompt 2.1) from prompts.md:
  1. Attention U-Net: Attention Gates on skip connections to filter sea clutter and focus on dark slicks.
  2. Residual U-Net: Residual connections across DoubleConv blocks for robust gradient flow and edge retention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# Attention Gate (AG)
# -------------------------------------------------------------------------
class AttentionGate(nn.Module):
    """
    Attention Gate for skip connections.
    Filters irrelevant background/ocean features and focuses activations on dark oil slick boundaries.
    Inputs:
      - g: Gating signal from the coarser (lower) level
      - x: Skip connection feature map from the encoder
    """
    def __init__(self, in_channels: int, gating_channels: int, inter_channels: int = None):
        super(AttentionGate, self).__init__()
        if inter_channels is None:
            inter_channels = in_channels // 2
            if inter_channels == 0:
                inter_channels = 1

        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Align spatial dimensions if necessary
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=True)

        theta_x = self.W_x(x)
        phi_g = self.W_g(g)
        f = self.relu(theta_x + phi_g)
        alpha = self.psi(f)
        return x * alpha


# -------------------------------------------------------------------------
# Residual Double Conv
# -------------------------------------------------------------------------
class ResidualDoubleConv(nn.Module):
    """
    DoubleConv with a residual shortcut:
      output = DoubleConv(x) + Shortcut(x)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))


class DownBlock(nn.Module):
    """Downscaling with MaxPool then Residual Double Conv."""
    def __init__(self, in_channels: int, out_channels: int):
        super(DownBlock, self).__init__()
        self.mp = nn.MaxPool2d(2)
        self.conv = ResidualDoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.mp(x))


class AttentionUpBlock(nn.Module):
    """
    Upscaling block equipped with an Attention Gate.
    1. Upsample lower features.
    2. Apply Attention Gate to skip features.
    3. Concatenate and apply Residual Double Conv.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super(AttentionUpBlock, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv_reduce = nn.Identity()

        self.ag = AttentionGate(in_channels=out_channels, gating_channels=in_channels)
        self.conv = ResidualDoubleConv(out_channels * 2, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1 is coarse gating signal, x2 is skip connection from encoder
        x1_up = self.up(x1)
        x1_up = self.conv_reduce(x1_up)

        # Pad if dimension mismatch
        diffY = x2.size()[2] - x1_up.size()[2]
        diffX = x2.size()[3] - x1_up.size()[3]
        if diffX > 0 or diffY > 0:
            x1_up = F.pad(x1_up, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Filter skip connection through attention gate
        x2_att = self.ag(g=x1, x=x2)

        x = torch.cat([x2_att, x1_up], dim=1)
        return self.conv(x)


# -------------------------------------------------------------------------
# Attention U-Net Model
# -------------------------------------------------------------------------
class AttentionUNet(nn.Module):
    """
    Attention U-Net for SAR Oil Spill Detection.
    Features:
      - Multi-scale Attention Gates suppressing ocean clutter.
      - Residual Double Convolution blocks.
      - Sigmoid probability output.
    """
    def __init__(self, n_channels: int = 1, n_classes: int = 1, bilinear: bool = True, base: int = 32):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base = base

        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16

        self.inc = ResidualDoubleConv(n_channels, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)
        self.down4 = DownBlock(c4, c5)

        self.up1 = AttentionUpBlock(c5, c4, bilinear=bilinear)
        self.up2 = AttentionUpBlock(c4, c3, bilinear=bilinear)
        self.up3 = AttentionUpBlock(c3, c2, bilinear=bilinear)
        self.up4 = AttentionUpBlock(c2, c1, bilinear=bilinear)

        self.outc = nn.Sequential(
            nn.Conv2d(c1, n_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    # Test model forward pass
    x = torch.randn(2, 1, 256, 256)
    model = AttentionUNet(n_channels=1, n_classes=1, base=32)
    y = model(x)
    print(f"AttentionUNet input: {x.shape}, output: {y.shape}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,} (~{num_params * 4 / (1024*1024):.2f} MB in float32)")
    print("AttentionUNet forward pass test passed successfully!")
