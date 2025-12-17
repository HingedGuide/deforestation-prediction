import torch
import torch.nn as nn

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


class ResidualBlock3D(nn.Module):
    """
    3D Residual Block:
    Input -> Conv3d -> BN3d -> ReLU -> Conv3d -> BN3d -> (+ Input) -> ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Check if stride is an integer or tuple
        # If integer, we apply it to spatial dims only (H, W) to preserve Time (T)
        # unless you specifically want to downsample time.
        if isinstance(stride, int):
            stride_tuple = (1, stride, stride)
        else:
            stride_tuple = stride

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride_tuple, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Shortcut to match dimensions if stride != 1 or channels change
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


class ResUNet3D(nn.Module):
    """
    3D ResUNet.
    Processes the input as a volume (B, C, T, H, W) using 3D convolutions.
    """

    def __init__(self, in_channels, time_depth, num_classes=2, filters=[32, 64, 128, 256]):
        super().__init__()

        # Encoder
        # Initial block
        self.input_layer = nn.Sequential(
            nn.Conv3d(in_channels, filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(filters[0], filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(inplace=True)
        )

        # Downsampling blocks using ResidualBlock3D
        # Stride is set to 2, which inside ResidualBlock3D maps to (1, 2, 2)
        # to downsample H and W but preserve T.
        self.res_down1 = ResidualBlock3D(filters[0], filters[1], stride=2)
        self.res_down2 = ResidualBlock3D(filters[1], filters[2], stride=2)
        self.res_down3 = ResidualBlock3D(filters[2], filters[3], stride=2)

        # Bridge (Bottleneck)
        self.bridge = ResidualBlock3D(filters[3], filters[3], stride=1)

        # Decoder
        # We use ConvTranspose3d with stride (1, 2, 2) to match the encoder's downsampling
        self.up3 = nn.ConvTranspose3d(filters[3], filters[2], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.res_up3 = ResidualBlock3D(filters[2] + filters[2], filters[2])

        self.up2 = nn.ConvTranspose3d(filters[2], filters[1], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.res_up2 = ResidualBlock3D(filters[1] + filters[1], filters[1])

        self.up1 = nn.ConvTranspose3d(filters[1], filters[0], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.res_up1 = ResidualBlock3D(filters[0] + filters[0], filters[0])

        # Final Classifier
        # Projects 3D features to classes. We still have the Time dimension here.
        self.classifier_3d = nn.Conv3d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        # x shape: [Batch, Channels, Time, Height, Width]

        # Encoder
        x1 = self.input_layer(x)  # [B, 32, T, H, W]
        x2 = self.res_down1(x1)  # [B, 64, T, H/2, W/2]
        x3 = self.res_down2(x2)  # [B, 128, T, H/4, W/4]
        x4 = self.res_down3(x3)  # [B, 256, T, H/8, W/8]

        # Bridge
        bridge = self.bridge(x4)

        # Decoder
        up3 = self.up3(bridge)  # Upsample spatial dims
        concat3 = torch.cat([up3, x3], dim=1)
        dec3 = self.res_up3(concat3)

        up2 = self.up2(dec3)
        concat2 = torch.cat([up2, x2], dim=1)
        dec2 = self.res_up2(concat2)

        up1 = self.up1(dec2)
        concat1 = torch.cat([up1, x1], dim=1)
        dec1 = self.res_up1(concat1)

        # 3D Logits: [B, NumClasses, T, H, W]
        logits_3d = self.classifier_3d(dec1)

        # Collapse Time Dimension for final 2D Prediction
        # We can take the mean over time or the last time step.
        # Here we use mean to aggregate temporal information.
        logits_2d = torch.mean(logits_3d, dim=2)  # [B, NumClasses, H, W]

        return logits_2d


class Simple3DCNN(nn.Module):
    """
    A simple 3D CNN baseline.
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


class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell.
    """

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


class ConvLSTM(nn.Module):
    """
    ConvLSTM Model for Spatio-Temporal Prediction.

    Architecture:
    1. Input (B, C, T, H, W) -> Unroll Time
    2. Pass through ConvLSTM Cell at each step
    3. Take the last Hidden State (B, Hidden, H, W)
    4. Pass through a Decoder/Classifier to get (B, NumClasses, H, W)
    """

    def __init__(self, in_channels, time_depth, num_classes=2, hidden_dim=64, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. Feature Extractor (Optional: Reduces input channels/noise before LSTM)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        # 2. ConvLSTM Layer
        self.lstm_cell = ConvLSTMCell(
            in_channels=32,
            hidden_channels=hidden_dim,
            kernel_size=kernel_size,
            bias=True
        )

        # 3. Final Classifier (1x1 Conv)
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x):
        # x: [Batch, Channels, Time, Height, Width]
        b, c, t, h, w = x.shape

        # Initialize hidden state
        hidden_state = self.lstm_cell.init_hidden(b, (h, w))

        # Loop over time steps
        for step in range(t):
            # Extract the slice for this time step: [B, C, H, W]
            x_t = x[:, :, step, :, :]

            # Encode features: [B, 32, H, W]
            x_t_encoded = self.encoder(x_t)

            # Update LSTM state
            hidden_state = self.lstm_cell(x_t_encoded, hidden_state)

        # Use the final hidden state (h_next) for prediction
        # h_n shape: [B, Hidden, H, W]
        final_h, _ = hidden_state

        logits = self.classifier(final_h)  # [B, NumClasses, H, W]
        return logits


class ViViTSegmentation(nn.Module):
    """
    Factorized Video Vision Transformer (ViViT) for Segmentation.

    This model treats the input as a video sequence (B, C, T, H, W).
    It uses a "Factorized Encoder" design:
      1. Tubelet Embedding: Projects 3D patches into tokens.
      2. Spatial Transformer: Processes spatial tokens for each frame independently.
      3. Temporal Transformer: Processes temporal evolution for each spatial location.
      4. Decoder: Projects the learned features back to a 2D segmentation mask.
    """

    def __init__(self, in_channels, time_depth, img_size=64, patch_size=8, embed_dim=128,
                 spatial_depth=4, temporal_depth=4, num_heads=4, num_classes=2):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # 1. Tubelet Embedding (3D Convolution)
        # We use kernel_size=(1, P, P) to treat each time step as a distinct frame of patches.
        # This preserves the T dimension exactly as is.
        self.tubelet_embed = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )

        # Calculate number of spatial patches
        self.num_patches_h = img_size // patch_size
        self.num_patches_w = img_size // patch_size
        self.num_spatial_tokens = self.num_patches_h * self.num_patches_w

        # Positional Embeddings
        # Spatial: (1, 1, N_s, E) - broadcasts across Time and Batch
        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, 1, self.num_spatial_tokens, embed_dim))
        # Temporal: (1, T, 1, E) - broadcasts across Spatial tokens and Batch
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, time_depth, 1, embed_dim))

        # 2. Spatial Transformer Encoder
        spatial_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=spatial_depth)

        # 3. Temporal Transformer Encoder
        temporal_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=temporal_depth)

        # 4. Decoder (Upsampling)
        # Projects the aggregated features back to the original image resolution.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # x shape: [Batch, C, Time, Height, Width]
        b, c, t, h, w = x.shape

        # --- A. Embedding ---
        # [B, Embed, T, H/P, W/P]
        x = self.tubelet_embed(x)

        # Reshape for Transformers:
        # We want to separate spatial and temporal dimensions.
        # Permute to [B, T, H_p*W_p, Embed]
        x = x.flatten(3).permute(0, 2, 3, 1)

        # Add Positional Embeddings
        # We slice pos_embed_temporal to support dynamic time lengths (RQ2) if t < max_time
        x = x + self.pos_embed_spatial + self.pos_embed_temporal[:, :t, :, :]

        # --- B. Spatial Transformer ---
        # Merge Batch and Time to process each frame's spatial tokens independently
        # Shape: [B * T, N_spatial, Embed]
        x_spatial = x.reshape(b * t, self.num_spatial_tokens, self.embed_dim)
        x_spatial = self.spatial_transformer(x_spatial)

        # --- C. Temporal Transformer ---
        # Reshape to separate Batch and Spatial, putting Time in the sequence dimension
        # Shape: [B * N_spatial, T, Embed]
        x_temporal = x_spatial.view(b, t, self.num_spatial_tokens, self.embed_dim).permute(0, 2, 1, 3)
        x_temporal = x_temporal.reshape(b * self.num_spatial_tokens, t, self.embed_dim)

        x_temporal = self.temporal_transformer(x_temporal)

        # --- D. Aggregation ---
        # We now have spatio-temporal features. We need to collapse Time to get a single prediction map.
        # Average Pooling over time is robust for summarizing the window.
        # Shape: [B * N_spatial, Embed]
        x_pooled = x_temporal.mean(dim=1)

        # --- E. Decoding ---
        # Reshape back to spatial grid: [B, Embed, H_p, W_p]
        x_2d = x_pooled.view(b, self.num_spatial_tokens, self.embed_dim).permute(0, 2, 1)
        x_2d = x_2d.view(b, self.embed_dim, self.num_patches_h, self.num_patches_w)

        logits = self.decoder(x_2d)  # [B, NumClasses, H, W]
        return logits