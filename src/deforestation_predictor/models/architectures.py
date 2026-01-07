import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A standard 2D Residual Block with Skip Connections.

    This block learns residual functions with reference to the layer input,
    which helps in training deeper networks by preventing the vanishing gradient problem.

    Structure:
        Input -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+ Input) -> ReLU

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride for the first convolution. Defaults to 1.
                                If stride > 1, the spatial dimensions are reduced.
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
        """
        Forward pass of the Residual Block.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, in_channels, Height, Width).

        Returns:
            torch.Tensor: Output tensor of shape (Batch, out_channels, Height/stride, Width/stride).
        """
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
    A 3D Residual Block for Volumetric or Spatiotemporal Data.

    Similar to the 2D version but uses Conv3d layers to process Time/Depth dimensions
    alongside spatial dimensions.

    Structure:
        Input -> Conv3d -> BN3d -> ReLU -> Conv3d -> BN3d -> (+ Input) -> ReLU

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int or tuple, optional): Stride for downsampling.
            - If int: applied as (1, stride, stride) to preserve the temporal dimension.
            - If tuple: applied directly as (d_stride, h_stride, w_stride).
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
        """
        Forward pass of the 3D Residual Block.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, in_channels, Time, Height, Width).

        Returns:
            torch.Tensor: Output tensor with transformed channels and potentially reduced spatial dimensions.
        """
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
    2D ResUNet for 3D Input (Early Fusion / "Flat" Time).

    This model flattens the temporal dimension into the channel dimension.
    For example, if input is 10 time steps with 3 bands, it processes it as a single
    2D image with 30 channels.

    Strategy:
        1. Flatten Time -> Channels.
        2. Encoder (Residual Blocks with downsampling).
        3. Bridge.
        4. Decoder (Upsampling + Skip Connections).

    Args:
        in_channels (int): Number of channels per time step.
        time_depth (int): Number of time steps in the input sequence.
        num_classes (int, optional): Number of output classes. Defaults to 2.
        filters (list, optional): List of channel counts for each encoder level. Defaults to [32, 64, 128, 256].
    """

    def __init__(self, in_channels, time_depth, num_classes=2, filters=[32, 64, 128, 256]):
        super().__init__()

        # Total input channels = variables * time_steps (Early Fusion)
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
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input shape [Batch, Channels, Time, Height, Width].

        Returns:
            torch.Tensor: Logits of shape [Batch, num_classes, Height, Width].
        """
        # x shape: [Batch, Channels, Time, Height, Width]
        b, c, t, h, w = x.shape

        # Flatten Time into Channels: [Batch, C*T, H, W]
        x = x.view(b, c * t, h, w)

        # Encoder
        x1 = self.input_layer(x)  # [B, 32, H, W]
        x2 = self.res_down1(x1)   # [B, 64, H/2, W/2]
        x3 = self.res_down2(x2)   # [B, 128, H/4, W/4]
        x4 = self.res_down3(x3)   # [B, 256, H/8, W/8]

        # Bridge
        bridge = self.bridge(x4)

        # Decoder (with skip connections)
        up3 = self.up3(bridge)    # [B, 128, H/4, W/4]
        concat3 = torch.cat([up3, x3], dim=1)
        dec3 = self.res_up3(concat3)

        up2 = self.up2(dec3)      # [B, 64, H/2, W/2]
        concat2 = torch.cat([up2, x2], dim=1)
        dec2 = self.res_up2(concat2)

        up1 = self.up1(dec2)      # [B, 32, H, W]
        concat1 = torch.cat([up1, x1], dim=1)
        dec1 = self.res_up1(concat1)

        logits = self.classifier(dec1)  # [B, num_classes, H, W]
        return logits


class ResUNet3D(nn.Module):
    """
    Full 3D ResUNet (Volumetric Convolution).

    Processes data as a 3D volume (Time, Height, Width).
    The Encoder downsamples spatial dimensions (H, W) but preserves Time (T).
    The Classifier collapses the Time dimension at the very end to produce a 2D map.

    Args:
        in_channels (int): Number of input channels.
        time_depth (int): The size of the temporal dimension (T).
        num_classes (int, optional): Number of output classes. Defaults to 2.
        filters (list, optional): List of channel counts. Defaults to [32, 64, 128, 256].
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
        # Collapses the temporal dimension (T) to 1.
        self.classifier_3d = nn.Conv3d(
            filters[0], 
            num_classes, 
            kernel_size=(time_depth, 1, 1), 
            padding=0
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input shape [Batch, Channels, Time, Height, Width].

        Returns:
            torch.Tensor: Logits of shape [Batch, num_classes, Height, Width].
        """
        # Encoder
        x1 = self.input_layer(x)  # [B, 32, T, H, W]
        x2 = self.res_down1(x1)   # [B, 64, T, H/2, W/2]
        x3 = self.res_down2(x2)   # [B, 128, T, H/4, W/4]
        x4 = self.res_down3(x3)   # [B, 256, T, H/8, W/8]

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

        # 3D Logits: [B, NumClasses, 1, H, W] (After kernel=(time_depth, 1, 1))
        logits_3d = self.classifier_3d(dec1)

        # Squeeze dimension 2 (Time) -> [B, NumClasses, H, W]
        logits_2d = logits_3d.squeeze(2)

        return logits_2d


class ConvLSTMCell(nn.Module):
    """
    A single Convolutional LSTM Cell.

    Performs the standard LSTM gating mechanisms (Input, Forget, Output, Gate)
    using Convolutional operations instead of matrix multiplications to preserve spatial structure.

    Args:
        in_channels (int): Input channel dimension.
        hidden_channels (int): Hidden state channel dimension.
        kernel_size (int): Convolution kernel size.
        bias (bool): Whether to use bias in convolutions.
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
        """
        Forward pass for one time step.

        Args:
            input_tensor (torch.Tensor): [Batch, in_channels, H, W]
            cur_state (tuple): (h_cur, c_cur), each of shape [Batch, hidden_channels, H, W]

        Returns:
            tuple: (h_next, c_next), the updated hidden and cell states.
        """
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
        """
        Initializes hidden states with zeros.

        Args:
            batch_size (int): Batch size.
            image_size (tuple): (Height, Width) of the feature map.

        Returns:
            tuple: (h, c) initialized to zeros.
        """
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device))


class ConvLSTM3D(nn.Module):
    """
    Hybrid Spatiotemporal Model: ResUNet Encoder + ConvLSTM Bridge + 2D Decoder.

    This architecture processes video data by:
    1. Encoder (Shared 2D): Extracts spatial features from each frame independently.
    2. Bridge (ConvLSTM): Processes the sequence of spatial features to model temporal evolution.
    3. Decoder (2D): Upsamples the *final* hidden state of the LSTM to produce the prediction.

    Args:
        in_channels (int): Input channels per frame.
        time_depth (int): Number of frames.
        num_classes (int, optional): Number of output classes. Defaults to 2.
        filters (list, optional): Channel counts. Defaults to [32, 64, 128, 256].
    """

    def __init__(self, in_channels, time_depth, num_classes=2, filters=[32, 64, 128, 256]):
        super().__init__()
        self.filters = filters

        # --- Shared Encoder (Applied to each frame) ---
        # Note: input_dim is just 'in_channels' (e.g. 10 bands), not C*T
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )

        self.res_down1 = ResidualBlock(filters[0], filters[1], stride=2)
        self.res_down2 = ResidualBlock(filters[1], filters[2], stride=2)
        self.res_down3 = ResidualBlock(filters[2], filters[3], stride=2)

        # --- The Bridge: ConvLSTM ---
        # Replaces the 2D/3D Conv Bridge.
        # It takes the deepest features (filters[3]) and outputs the same size.
        self.lstm_bridge = ConvLSTMCell(
            in_channels=filters[3],
            hidden_channels=filters[3],
            kernel_size=3,
            bias=True
        )

        # --- Decoder (Standard 2D) ---
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.res_up3 = ResidualBlock(filters[2] + filters[2], filters[2])

        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.res_up2 = ResidualBlock(filters[1] + filters[1], filters[1])

        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.res_up1 = ResidualBlock(filters[0] + filters[0], filters[0])

        self.classifier = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input shape [Batch, Channels, Time, Height, Width].

        Returns:
            torch.Tensor: Logits of shape [Batch, num_classes, Height, Width].
        """
        # x shape: [Batch, Channels, Time, Height, Width]
        b, c, t, h, w = x.shape

        # 1. Rearrange to process all time steps at once through the 2D Encoder
        # Merge Batch and Time: [B*T, C, H, W]
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        # --- Encoder (Time-Distributed) ---
        # We save the features for each step, but we only strictly need the
        # features from the LAST step for the skip connections in a standard setup.
        e1 = self.input_layer(x_reshaped)  # [B*T, 32, H, W]
        e2 = self.res_down1(e1)  # [B*T, 64, H/2, W/2]
        e3 = self.res_down2(e2)  # [B*T, 128, H/4, W/4]
        e4 = self.res_down3(e3)  # [B*T, 256, H/8, W/8]

        # 2. Reshape back to Sequence for the LSTM Bridge
        # [B, T, 256, H/8, W/8]
        lstm_in = e4.view(b, t, self.filters[3], h // 8, w // 8)

        # --- Bridge (ConvLSTM) ---
        # Initialize hidden state
        h_state, c_state = self.lstm_bridge.init_hidden(b, (h // 8, w // 8))

        # Loop through time
        for step in range(t):
            # Input: [B, 256, H/8, W/8]
            step_input = lstm_in[:, step, :, :, :]
            h_state, c_state = self.lstm_bridge(step_input, (h_state, c_state))

        # We use the FINAL hidden state (h_state) as the bridge output
        bridge_out = h_state  # [B, 256, H/8, W/8]

        # --- Decoder ---
        # For skip connections, we use the features from the LAST time step (t-1).
        # We extract them from the reshaped encoder outputs.
        # e3 view: [B, T, 128, H/4, W/4] -> Take last T

        skip3 = e3.view(b, t, self.filters[2], h // 4, w // 4)[:, -1, ...]
        skip2 = e2.view(b, t, self.filters[1], h // 2, w // 2)[:, -1, ...]
        skip1 = e1.view(b, t, self.filters[0], h, w)[:, -1, ...]

        # Up 3
        up3 = self.up3(bridge_out)
        concat3 = torch.cat([up3, skip3], dim=1)
        dec3 = self.res_up3(concat3)

        # Up 2
        up2 = self.up2(dec3)
        concat2 = torch.cat([up2, skip2], dim=1)
        dec2 = self.res_up2(concat2)

        # Up 1
        up1 = self.up1(dec2)
        concat1 = torch.cat([up1, skip1], dim=1)
        dec1 = self.res_up1(concat1)

        logits = self.classifier(dec1)
        return logits


class ViViTSegmentation(nn.Module):
    """
    Factorized Video Vision Transformer (ViViT) for Segmentation.

    This model treats video as a sequence of tubelets (3D patches).
    It uses a "Factorized Encoder" approach:
    1. Spatial Transformer: Models relationships within a frame.
    2. Temporal Transformer: Models relationships across frames.

    It includes a 'Progressive Decoder' to upsample the low-res Transformer
    features back to the original image resolution.

    Args:
        in_channels (int): Input channels.
        time_depth (int): Number of frames.
        img_size (int): Spatial height/width of input (assumed square).
        patch_size (int): Size of the tubelet patch.
        embed_dim (int): Dimension of internal embeddings.
        spatial_depth (int): Number of Spatial Transformer layers.
        temporal_depth (int): Number of Temporal Transformer layers.
        num_heads (int): Number of attention heads.
        num_classes (int): Number of output classes.
    """

    def __init__(self, in_channels, time_depth, img_size=64, patch_size=8, embed_dim=128,
                 spatial_depth=4, temporal_depth=4, num_heads=4, num_classes=2):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # 1. Tubelet Embedding
        self.tubelet_embed = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )

        # Spatial Tokens
        self.num_patches_h = img_size // patch_size
        self.num_patches_w = img_size // patch_size
        self.num_spatial_tokens = self.num_patches_h * self.num_patches_w

        # Positional Embeddings
        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, 1, self.num_spatial_tokens, embed_dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, time_depth, 1, embed_dim))

        # 2. Spatial Transformer
        spatial_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=spatial_depth)

        # 3. Temporal Transformer
        temporal_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=temporal_depth)

        # 4. Progressive Decoder (Updated)
        # Instead of one big 8x jump, we do 2x -> 2x -> 2x.
        # This makes the output smoothness comparable to the CNN U-Nets.

        # Block 1: Upsample 8x8 (features) -> 16x16
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Block 2: Upsample 16x16 -> 32x32
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Block 3: Upsample 32x32 -> 64x64 (Original Size)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input shape [Batch, Channels, Time, Height, Width].

        Returns:
            torch.Tensor: Logits of shape [Batch, num_classes, Height, Width].
        """
        # x shape: [Batch, C, Time, Height, Width]
        b, c, t, h, w = x.shape

        # --- A. Embedding ---
        # Extract Tubelets and Flatten
        x = self.tubelet_embed(x)
        x = x.flatten(3).permute(0, 2, 3, 1)  # [B, T, N_spatial, Embed]
        # Add Positional Embeddings
        x = x + self.pos_embed_spatial + self.pos_embed_temporal[:, :t, :, :]

        # --- B. Spatial Transformer ---
        # Merge Batch and Time to process each frame spatially
        x_spatial = x.reshape(b * t, self.num_spatial_tokens, self.embed_dim)
        x_spatial = self.spatial_transformer(x_spatial)

        # --- C. Temporal Transformer ---
        # Reshape to group spatial tokens across time: [B * N_spatial, T, Embed]
        x_temporal = x_spatial.view(b, t, self.num_spatial_tokens, self.embed_dim).permute(0, 2, 1, 3)
        x_temporal = x_temporal.reshape(b * self.num_spatial_tokens, t, self.embed_dim)
        x_temporal = self.temporal_transformer(x_temporal)

        # --- D. Aggregation ---
        # Mean pool over time to get a single descriptor per spatial token
        x_pooled = x_temporal.mean(dim=1)  # [B * N_spatial, Embed]

        # --- E. Progressive Decoding ---
        # Reshape back to spatial grid: [B, Embed, H_p, W_p]
        x_2d = x_pooled.view(b, self.num_spatial_tokens, self.embed_dim).permute(0, 2, 1)
        x_2d = x_2d.view(b, self.embed_dim, self.num_patches_h, self.num_patches_w)

        # Upsample progressively
        x = self.dec1(x_2d)  # -> 16x16
        x = self.dec2(x)     # -> 32x32
        x = self.dec3(x)     # -> 64x64

        logits = self.classifier(x)
        return logits