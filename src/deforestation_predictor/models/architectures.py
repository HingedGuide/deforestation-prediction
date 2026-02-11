import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
#  BUILDING BLOCKS
# ==============================================================================

class ResidualBlock(nn.Module):
    """
    Standard 2D Residual Block (Used in ConvLSTM3D decoder).
    Structure: Input -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+ Input) -> ReLU
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

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

class ResidualConv(nn.Module):
    """
    Residual Convolution Block (Pre-activation).
    Used in Laura's ResUNet implementation.
    Structure: BN -> ReLU -> Conv -> BN -> ReLU -> Conv (+ Skip Conv)
    """
    def __init__(self, input_dim, kernel_size, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)

class Upsample_(nn.Module):
    """
    Wrapper for nn.Upsample to match Supervisor's structure (Bilinear).
    """
    def __init__(self, scale_factor=2.0):
        super(Upsample_, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

    def forward(self, x):
        return self.upsample(x)

class ResidualBlock3D(nn.Module):
    """
    A 3D Residual Block for Volumetric Data (Used in ResUNet3D and ConvLSTM3D).
    Structure: Input -> Conv3d -> BN3d -> ReLU -> Conv3d -> BN3d -> (+ Input) -> ReLU
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Handle stride tuple vs int
        if isinstance(stride, int):
            stride_tuple = (1, stride, stride) # Preserve time dim
        else:
            stride_tuple = stride

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride_tuple, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride_tuple, bias=False),
                nn.BatchNorm3d(out_channels)
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

# ==============================================================================
#  MODELS
# ==============================================================================

class ResUNet(nn.Module):
    """
    The 2D ResUNet (Supervisor's version)
    """
    def __init__(self, in_channels=33, num_classes=1, filters=[32, 64, 128, 256]):
        super(ResUNet, self).__init__()

        # --- Encoder ---
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, stride=1, padding=1)
        )
        
        self.residual_conv_1 = ResidualConv(filters[0], 3, filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], 3, filters[2], 2, 1)

        # --- Bridge ---
        self.bridge = ResidualConv(filters[2], 3, filters[3], 2, 1)

        # --- Decoder ---
        self.upsample_1 = Upsample_(scale_factor=2.0)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], 3, filters[2], 1, 1)
        
        self.upsample_2 = Upsample_(scale_factor=2.0)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], 3, filters[1], 1, 1)
        
        self.upsample_3 = Upsample_(scale_factor=2.0)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], 3, filters[0], 1, 1)

        self.drop = nn.Dropout2d(p=0.3)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], num_classes, 1, 1),
        )

    def forward(self, x):
        # --- Encode ---
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)

        # --- Bridge ---
        x4 = self.bridge(x3) 
        
        # --- Decode ---
        # Up 1
        x4_up = self.upsample_1(x4)
        if x4_up.size()[2:] != x3.size()[2:]:
            x4_up = F.interpolate(x4_up, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x5 = torch.cat([x4_up, x3], dim=1)
        x6 = self.up_residual_conv1(x5)

        # Up 2
        x6_up = self.upsample_2(x6)
        if x6_up.size()[2:] != x2.size()[2:]:
            x6_up = F.interpolate(x6_up, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x7 = torch.cat([x6_up, x2], dim=1)
        x8 = self.up_residual_conv2(x7)

        # Up 3
        x8_up = self.upsample_3(x8)
        if x8_up.size()[2:] != x1.size()[2:]:
            x8_up = F.interpolate(x8_up, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x9 = torch.cat([x8_up, x1], dim=1)
        x10 = self.up_residual_conv3(x9)
        
        # Output
        x10 = self.drop(x10)
        output = self.output_layer(x10)
        
        return output

class ResUNet3D(nn.Module):
    """
    Full 3D ResUNet (Volumetric Convolution).
    """
    def __init__(self, in_channels, time_depth, num_classes=2, filters=[32, 64, 128, 256]):
        super().__init__()

        # Encoder
        self.input_layer = nn.Sequential(
            nn.Conv3d(in_channels, filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(filters[0], filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(inplace=True)
        )

        self.res_down1 = ResidualBlock3D(filters[0], filters[1], stride=2)
        self.res_down2 = ResidualBlock3D(filters[1], filters[2], stride=2)
        self.res_down3 = ResidualBlock3D(filters[2], filters[3], stride=2)

        self.bridge = ResidualBlock3D(filters[3], filters[3], stride=1)

        # Decoder (Using ConvTranspose3d)
        self.up3 = nn.ConvTranspose3d(filters[3], filters[2], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.res_up3 = ResidualBlock3D(filters[2] + filters[2], filters[2])

        self.up2 = nn.ConvTranspose3d(filters[2], filters[1], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.res_up2 = ResidualBlock3D(filters[1] + filters[1], filters[1])

        self.up1 = nn.ConvTranspose3d(filters[1], filters[0], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.res_up1 = ResidualBlock3D(filters[0] + filters[0], filters[0])

        self.classifier_3d = nn.Conv3d(
            filters[0], num_classes, kernel_size=(time_depth, 1, 1), padding=0
        )

    def forward(self, x):
        # x shape: [Batch, Channels, Time, Height, Width]
        x1 = self.input_layer(x)
        x2 = self.res_down1(x1)
        x3 = self.res_down2(x2)
        x4 = self.res_down3(x3)

        bridge = self.bridge(x4)

        up3 = self.up3(bridge)
        concat3 = torch.cat([up3, x3], dim=1)
        dec3 = self.res_up3(concat3)

        up2 = self.up2(dec3)
        concat2 = torch.cat([up2, x2], dim=1)
        dec2 = self.res_up2(concat2)

        up1 = self.up1(dec2)
        concat1 = torch.cat([up1, x1], dim=1)
        dec1 = self.res_up1(concat1)

        logits_3d = self.classifier_3d(dec1)
        logits_2d = logits_3d.squeeze(2)

        return logits_2d

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, bias):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.in_channels + self.hidden_channels,
            out_channels=4 * self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device))

class ConvLSTM3D(nn.Module):
    """
    Hybrid Model: 3D Encoder -> ConvLSTM Bridge -> 2D Decoder
    """
    def __init__(self, in_channels, time_depth, num_classes=2, filters=[32, 64, 128, 256]):
        super().__init__()
        self.filters = filters

        self.input_layer = nn.Sequential(
            nn.Conv3d(in_channels, filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(filters[0], filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(inplace=True)
        )

        self.res_down1 = ResidualBlock3D(filters[0], filters[1], stride=2)
        self.res_down2 = ResidualBlock3D(filters[1], filters[2], stride=2)
        self.res_down3 = ResidualBlock3D(filters[2], filters[3], stride=2)

        self.lstm_bridge = ConvLSTMCell(
            in_channels=filters[3],
            hidden_channels=filters[3],
            kernel_size=3,
            bias=True
        )

        # Decoder uses standard 2D components
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.res_up3 = ResidualBlock(filters[2] + filters[2], filters[2])

        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.res_up2 = ResidualBlock(filters[1] + filters[1], filters[1])

        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.res_up1 = ResidualBlock(filters[0] + filters[0], filters[0])

        self.classifier = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        b, c, t, h, w = x.shape

        e1 = self.input_layer(x)
        e2 = self.res_down1(e1)
        e3 = self.res_down2(e2)
        e4 = self.res_down3(e3)

        lstm_in = e4.permute(0, 2, 1, 3, 4) 
        h_state, c_state = self.lstm_bridge.init_hidden(b, (h // 8, w // 8))

        for step in range(t):
            step_input = lstm_in[:, step, :, :, :]
            h_state, c_state = self.lstm_bridge(step_input, (h_state, c_state))

        bridge_out = h_state 

        skip3 = e3[:, :, -1, :, :]
        skip2 = e2[:, :, -1, :, :]
        skip1 = e1[:, :, -1, :, :]

        up3 = self.up3(bridge_out)
        concat3 = torch.cat([up3, skip3], dim=1)
        dec3 = self.res_up3(concat3)

        up2 = self.up2(dec3)
        concat2 = torch.cat([up2, skip2], dim=1)
        dec2 = self.res_up2(concat2)

        up1 = self.up1(dec2)
        concat1 = torch.cat([up1, skip1], dim=1)
        dec1 = self.res_up1(concat1)

        logits = self.classifier(dec1)
        return logits

class ViViTSegmentation(nn.Module):
    """
    Video Vision Transformer
    """
    def __init__(self, in_channels, time_depth, img_size=64, patch_size=8, embed_dim=128,
                 spatial_depth=4, temporal_depth=4, num_heads=4, num_classes=2):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.tubelet_embed = nn.Conv3d(
            in_channels, embed_dim, kernel_size=(1, patch_size, patch_size), stride=(1, patch_size, patch_size)
        )

        self.num_patches_h = img_size // patch_size
        self.num_patches_w = img_size // patch_size
        self.num_spatial_tokens = self.num_patches_h * self.num_patches_w

        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, 1, self.num_spatial_tokens, embed_dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, time_depth, 1, embed_dim))

        spatial_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=spatial_depth)

        temporal_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=temporal_depth)

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        b, c, t, h, w = x.shape

        x = self.tubelet_embed(x)
        x = x.flatten(3).permute(0, 2, 3, 1)
        x = x + self.pos_embed_spatial + self.pos_embed_temporal[:, :t, :, :]

        x_spatial = x.reshape(b * t, self.num_spatial_tokens, self.embed_dim)
        x_spatial = self.spatial_transformer(x_spatial)

        x_temporal = x_spatial.view(b, t, self.num_spatial_tokens, self.embed_dim).permute(0, 2, 1, 3)
        x_temporal = x_temporal.reshape(b * self.num_spatial_tokens, t, self.embed_dim)
        x_temporal = self.temporal_transformer(x_temporal)

        x_pooled = x_temporal.mean(dim=1)

        x_2d = x_pooled.view(b, self.num_spatial_tokens, self.embed_dim).permute(0, 2, 1)
        x_2d = x_2d.view(b, self.embed_dim, self.num_patches_h, self.num_patches_w)

        x = self.dec1(x_2d)
        x = self.dec2(x)
        x = self.dec3(x)

        logits = self.classifier(x)
        return logits