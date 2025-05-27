import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Residual Block with Dilation
# -------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout_rate)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

# ----------------------------------------
#  Multi-Scale Feature Extractor
# ----------------------------------------
class MultiScaleBlock(nn.Module):
    def __init__(self, channels, dilations=[1, 2, 4]):
        super(MultiScaleBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=d, dilation=d, bias=False) for d in dilations
        ])
        self.bn = nn.BatchNorm2d(channels * len(dilations))
        self.relu = nn.ReLU(inplace=True)
        self.conv_fuse = nn.Conv2d(channels * len(dilations), channels, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        outs = [conv(x) for conv in self.convs]
        out = torch.cat(outs, dim=1)
        out = self.relu(self.bn(out))
        out = self.relu(self.bn_fuse(self.conv_fuse(out)))
        return out

# -----------------------------
# Channel Attention Mechanism
# -----------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# -----------------------------
# Spatial Attention Mechanism
# -----------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

# ---------------------
# Non-local Block
# ---------------------
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        # Reduce channel dimensionality for efficiency
        self.inter_channels = in_channels // 2 if in_channels > 1 else 1
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        batch_size, C, H, W = x.shape
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # (B, inter, N)
        g_x = g_x.permute(0, 2, 1)  # (B, N, inter)
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # (B, inter, N)
        theta_x = theta_x.permute(0, 2, 1)  # (B, N, inter)
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # (B, inter, N)
        
        f = torch.matmul(theta_x, phi_x)  # (B, N, N)
        f_div_C = f / (H * W)
        
        y = torch.matmul(f_div_C, g_x)  # (B, N, inter)
        y = y.permute(0, 2, 1).contiguous()  # (B, inter, N)
        y = y.view(batch_size, self.inter_channels, H, W)
        y = self.conv_out(y)
        y = self.bn(y)
        
        out = x + y
        return out

# ---------------------
# GomokuNet Definition
# ---------------------
class GomokuNet(nn.Module):
    def __init__(self, board_size=15, in_channels=3, num_classes=225):
        """
        Args:
            board_size (int): The board dimensions (e.g., 15 for 15x15).
            in_channels (int): Number of input channels (3 in this case).
            num_classes (int): Number of output classes (225 move positions).
        """
        super(GomokuNet, self).__init__()
        self.board_size = board_size
        
        # Initial convolution to raise channel dimension
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05)
        )
        
        # Residual blocks with dilation and dropout
        self.resblock1 = ResidualBlock(64, dilation=1, dropout_rate=0.1)
        self.resblock2 = ResidualBlock(64, dilation=2, dropout_rate=0.125)
        self.resblock3 = ResidualBlock(64, dilation=4, dropout_rate=0.15)
        
        # Optional Multi-scale feature extractor
        self.multi_scale = MultiScaleBlock(64, dilations=[1, 2, 4])
        
        # Attention mechanisms (channel + spatial)
        self.channel_attention = ChannelAttention(64)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
        # Non-local block for long-range patterns
        self.non_local = NonLocalBlock(64)
        
        # Policy head: reduce channels and flatten for classification
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15)
        )
        # Fully connected layer to produce logits for 225 board positions
        self.policy_fc = nn.Sequential(
            nn.Linear(32 * board_size * board_size, num_classes),
            nn.Dropout(0.1)
        )
        
        # Value head: predicts win probability for current player
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15)
        )
        # Value tower - fully connected layers
        self.value_fc = nn.Sequential(
            nn.Linear(32 * board_size * board_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.125),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Tanh()  # Output in range [-1, 1], where 1 means 100% win for current player
        )
        
    def forward(self, x):
        """
        Forward pass.
        Input:
            x: Tensor of shape (B, 3, board_size, board_size)
        Output:
            policy_logits: Tensor of shape (B, num_classes)
            value: Tensor of shape (B, 1) with win probability estimate
        """
        x = self.initial_conv(x)  # (B, 64, board_size, board_size)
        
        # Apply residual blocks with increasing dilation
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        
        # Multi-scale feature extraction
        x = self.multi_scale(x)
        
        # Apply attention: channel and spatial
        ca = self.channel_attention(x)
        sa = self.spatial_attention(x)
        x = x * ca * sa
        
        # Incorporate non-local dependencies
        x = self.non_local(x)
        
        # Policy head and flatten for classification
        policy = self.policy_head(x)  # (B, 32, board_size, board_size)
        policy = policy.view(policy.size(0), -1)  # flatten to (B, 32 * board_size * board_size)
        policy_logits = self.policy_fc(policy)  # (B, num_classes)
        
        # Value head
        value = self.value_head(x)  # (B, 32, board_size, board_size)
        value = value.view(value.size(0), -1)  # flatten to (B, 32 * board_size * board_size)
        value = self.value_fc(value)  # (B, 1)
        
        return policy_logits, value


if __name__ == "__main__":
    model = GomokuNet()
    sample_input = torch.randn(2, 3, 15, 15)  # Batch of 2 samples
    policy_output, value_output = model(sample_input)
    print("Policy output shape:", policy_output.shape)  # Expected: (2, 225)
    print("Value output shape:", value_output.shape)    # Expected: (2, 1)
