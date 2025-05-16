import torch
import torch.nn as nn
import torch.nn.functional as F

class ResDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 with residual connection"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        # Main branch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Residual connection (with 1x1 conv if dimensions don't match)
        self.residual = nn.Sequential()
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Final ReLU after addition
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.residual(x)
        out = self.double_conv(x)
        out += identity  # Add residual connection
        out = self.relu(out)  # Apply ReLU after addition
        return out


class ResDown(nn.Module):
    """Downscaling with maxpool then residual double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.res_conv = ResDoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        return self.res_conv(x)


class ResUp(nn.Module):
    """Upscaling then residual double conv"""

    def __init__(self, in_channels_up, in_channels_skip, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use a simple upsampling layer and then a conv to adjust channels
        # else, use a learnable transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After upsampling, in_channels_up remains. After cat with in_channels_skip, it's in_channels_up + in_channels_skip.
            self.res_conv = ResDoubleConv(in_channels_up + in_channels_skip, out_channels)
        else:
            # ConvTranspose2d typically halves the input channels (in_channels_up)
            self.up = nn.ConvTranspose2d(in_channels_up, in_channels_up // 2, kernel_size=2, stride=2)
            # After upsampling, channels are in_channels_up // 2. After cat with in_channels_skip, it's in_channels_up // 2 + in_channels_skip.
            self.res_conv = ResDoubleConv(in_channels_up // 2 + in_channels_skip, out_channels)


    def forward(self, x1_up, x2_skip):
        """
        x1_up: tensor from the upsampling path (lower layer in U-Net)
        x2_skip: tensor from the skip connection (encoder path)
        """
        x1_up = self.up(x1_up) # Upsample x1

        # Pad x1_up to match spatial dimensions of x2_skip if they differ
        diffY = x2_skip.size()[2] - x1_up.size()[2]
        diffX = x2_skip.size()[3] - x1_up.size()[3]

        # F.pad expects (padding_left, padding_right, padding_top, padding_bottom)
        x1_up = F.pad(x1_up, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2_skip, x1_up], dim=1) # Concatenate along channel dimension
        return self.res_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResUNet(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, bilinear=True):
        super(ResUNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear

        # Encoder with residual blocks
        self.inc = ResDoubleConv(n_channels_in, 64)     # x_skip1 (H, W)
        self.down1 = ResDown(64, 128)                   # x_skip2 (H/2, W/2)
        self.down2 = ResDown(128, 256)                  # x_skip3 (H/4, W/4)
        self.down3 = ResDown(256, 512)                  # x_skip4 (H/8, W/8)
        
        # Bottleneck - using fewer channels and a single bottleneck for efficiency
        self.bottleneck = ResDoubleConv(512, 512)       # Bottleneck (H/8, W/8) - reduced from 1024 to 512
        
        # Decoder with residual blocks
        # More efficient channel count progression
        self.up1 = ResUp(512, 512, 256, bilinear)      # Upsamples 512 (from bottleneck), skips 512 (from down3)
        self.up2 = ResUp(256, 256, 128, bilinear)      # Upsamples 256 (from up1), skips 256 (from down2)
        self.up3 = ResUp(128, 128, 64, bilinear)       # Upsamples 128 (from up2), skips 128 (from down1)
        self.up4 = ResUp(64, 64, 64, bilinear)         # Upsamples 64 (from up3), skips 64 (from inc)
        
        self.outc = OutConv(64, n_channels_out)

    def forward(self, x):
        # Encoder
        x_skip1 = self.inc(x)
        x_skip2 = self.down1(x_skip1)
        x_skip3 = self.down2(x_skip2)
        x_skip4 = self.down3(x_skip3)
        bottleneck = self.bottleneck(x_skip4)

        # Decoder
        x = self.up1(bottleneck, x_skip4)
        x = self.up2(x, x_skip3)
        x = self.up3(x, x_skip2)
        x = self.up4(x, x_skip1)
        
        logits = self.outc(x)
        return logits


# Additional memory-saving options (can be enabled as needed)
def add_memory_efficient_options(model):
    """
    Adds memory-efficient options to the model for training on limited GPU resources.
    """
    # Option to use gradient checkpointing to save memory (at the cost of some computation)
    if hasattr(model, 'use_gradient_checkpointing'):
        def enable_gradient_checkpointing(self):
            self.apply(lambda module: setattr(module, 'gradient_checkpointing', True))
        type(model).enable_gradient_checkpointing = enable_gradient_checkpointing
    
    return model


# Example of creating a memory-efficient ResUNet
def create_efficient_resunet(n_channels_in, n_channels_out, bilinear=True):
    model = ResUNet(n_channels_in, n_channels_out, bilinear)
    return add_memory_efficient_options(model)

# Usage example:
# model = create_efficient_resunet(3, 1)  # 3 input channels (RGB), 1 output channel (segmentation)