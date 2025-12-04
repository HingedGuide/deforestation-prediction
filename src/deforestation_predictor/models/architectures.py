import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Standard Residual Block:
    Input -> Conv -> BN -> ReLU -> Conv -> BN -> (+ Input) -> ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut to match dimensions if stride != 1 or channels change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ResUNet(nn.Module):
    """
    ResUNet for 3D Input (Time is flattened into Channels).
    """

    def __init__(self, in_channels, time_depth, num_classes=2, filters=[32, 64, 128, 256]):
        super().__init__()

        # Total input channels = variables * time_steps
        input_dim = in_channels * time_depth

        # Encoder
        self.input_layer = nn.Sequential(
            nn.Conv2d(input_dim, filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )

        self.res_down1 = ResidualBlock(filters[0], filters[1], stride=2)
        self.res_down2 = ResidualBlock(filters[1], filters[2], stride=2)
        self.res_down3 = ResidualBlock(filters[2], filters[3], stride=2)

        # Bridge
        self.bridge = ResidualBlock(filters[3], filters[3], stride=1)

        # Decoder
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.res_up3 = ResidualBlock(filters[2] + filters[2], filters[2])  # + filters[2] for concat

        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.res_up2 = ResidualBlock(filters[1] + filters[1], filters[1])

        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.res_up1 = ResidualBlock(filters[0] + filters[0], filters[0])

        self.classifier = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        # x shape: [Batch, Channels, Time, Height, Width]
        b, c, t, h, w = x.shape

        # Flatten Time into Channels: [Batch, C*T, H, W]
        x = x.view(b, c * t, h, w)

        # Encoder
        x1 = self.input_layer(x)  # [B, 32, H, W]
        x2 = self.res_down1(x1)  # [B, 64, H/2, W/2]
        x3 = self.res_down2(x2)  # [B, 128, H/4, W/4]
        x4 = self.res_down3(x3)  # [B, 256, H/8, W/8]

        # Bridge
        bridge = self.bridge(x4)

        # Decoder (with skip connections)
        up3 = self.up3(bridge)  # [B, 128, H/4, W/4]
        concat3 = torch.cat([up3, x3], dim=1)
        dec3 = self.res_up3(concat3)

        up2 = self.up2(dec3)  # [B, 64, H/2, W/2]
        concat2 = torch.cat([up2, x2], dim=1)
        dec2 = self.res_up2(concat2)

        up1 = self.up1(dec2)  # [B, 32, H, W]
        concat1 = torch.cat([up1, x1], dim=1)
        dec1 = self.res_up1(concat1)

        logits = self.classifier(dec1)  # [B, num_classes, H, W]
        return logits


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