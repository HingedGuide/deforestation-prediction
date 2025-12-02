import torch
import torch.nn as nn


class Simple3DCNN(nn.Module):
    """
    A simple 3D CNN baseline for RQ1.
    Treats time as the depth dimension (D) in (N, C, D, H, W).
    """

    def __init__(self, in_channels, time_depth, num_classes=2):
        super().__init__()

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2))  # Downsample T, H, W
        )

        # Decoder (Simplified for example; in real usage, use TransposeConv or Upsample)
        # We need to flatten Time and project back to 2D Spatial mask
        self.final_conv = nn.Conv2d(64 * (time_depth // 2), 64, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # x: [Batch, Channels, Time, Height, Width]
        x = self.conv1(x)
        x = self.conv2(x)

        # Flatten Time dimension into Channels for 2D prediction
        b, c, t, h, w = x.shape
        x = x.view(b, c * t, h, w)

        # Upsample back to original spatial size (since we pooled by 2)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.final_conv(x)
        logits = self.classifier(x)  # [Batch, NumClasses, H, W]
        return logits

# Note: You can add a ResUNet3D or ConvLSTM class here following the same structure.