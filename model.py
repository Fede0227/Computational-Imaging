import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block
# This is the core module of the architecture. It has two paths:
# One that applies convolutions and normalization
# One that does nothing 
# The model learns how much to use either path, potentially it can also "do nothing"
#the residual is implemented via sum at the end 
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

# Downsampling block
# This reduces the spatial resolution and increases the feature size
# It uses stride=2 to downsample and includes a residual block for learning
# If this is not clear you need to understand better how convolution works (strides in particular)
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),  # halve dimensions
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )

    def forward(self, x):
        return self.down(x)

# Upsampling block
# These are the mirror of down blocks. They:
# Use bilinear upsampling to increase the spatial dimension instead of stride = 2 (we could use transposed convolution but we keep it simple)
# Concatenate skip connections from the encoder
# Use a 1x1 conv to reduce the number of channels after concatenation
# Followed by a residual block to learn
class UpBlock(nn.Module):
    # We have a different signature because we need to account for skip connections
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.res_block = ResidualBlock(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1x1(x)
        return self.res_block(x)

# Residual U-Net
# This combines the blocks above into the full encoder-bottleneck-decoder architecture
class ResidualUNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[32, 64, 128, 256]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Initial conv before the encoder
        # This is done to match the input channels to the initial number of filters
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(features[0])
        )

        # Build encoder: progressively reduce spatial dimensions and increase features
        for i in range(len(features) - 1):
            self.encoder.append(DownBlock(features[i], features[i+1]))

        # Bottleneck: deepest part of the network, just res block
        self.bottleneck = ResidualBlock(features[-1])

        # Build decoder: upsample and reduce features
        # a little more complexity because we need to account for skip shape addition
        reversed_features = features[::-1]
        for i in range(len(reversed_features) - 1):
            skip_ch = reversed_features[i]
            out_ch = reversed_features[i + 1]
            in_ch = skip_ch + out_ch  # input to UpBlock = skip + upsampled decoder
            self.decoder.append(UpBlock(in_ch, out_ch))

        # Final output layer: 1x1 conv with zero initialization and no activation (for stability)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x):
        skips = []
        x = self.initial(x)
        skips.append(x)

        for down in self.encoder:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x)
        skips = skips[:-1][::-1]  # remove bottleneck skip and reverse

        for up, skip in zip(self.decoder, skips):
            x = up(x, skip)

        return self.final_conv(x)
