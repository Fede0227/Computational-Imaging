import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def plot_dataset_sample(era5_sample, vhr_sample):
    # Magnitude M = sqrt(u^2 + v^2)
    # PyTorch's torch.sqrt and element-wise operations make this easy.
    era5_magnitude = torch.sqrt(era5_sample[0, :, :]**2 + era5_sample[1, :, :]**2)
    vhr_magnitude = torch.sqrt(vhr_sample[0, :, :]**2 + vhr_sample[1, :, :]**2)

    # Matplotlib typically works best with NumPy arrays.
    era5_magnitude_numpy = era5_magnitude.cpu().numpy()
    vhr_magnitude_numpy = vhr_magnitude.cpu().numpy()

    plt.figure(figsize=(10, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(era5_magnitude_numpy, cmap='viridis', origin='lower')
    plt.title('Vector Field Intensity Map')
    plt.xlabel('Pixel X-coordinate (width)')
    plt.ylabel('Pixel Y-coordinate (height)')

    plt.subplot(1, 2, 2)
    plt.imshow(vhr_magnitude_numpy, cmap='viridis', origin='lower')
    plt.title('Vector Field Intensity Map')
    plt.xlabel('Pixel X-coordinate (width)')
    plt.ylabel('Pixel Y-coordinate (height)')

    plt.tight_layout()
    plt.show()


class ItalyWeatherDataset(Dataset):
    
    era5_variables = ["u10", "v10"]
    vhr_variables = ["U_10M", "V_10M"]
    
    def __init__(self, era5_dataset, vhr_dataset , transform=None, target_transform=None):
        super().__init__()
        self.era5_dataset = era5_dataset
        self.vhr_dataset = vhr_dataset
        self.transform = transform
        self.target_transform = target_transform

        self.num_samples = len(self.era5_dataset.valid_time)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not 0 <= idx < self.num_samples:
            raise IndexError(f"Index {idx} is out of bounds for dataset with size {self.num_samples}")

        era_sample_slice = self.era5_dataset.isel(valid_time=idx)
        vhr_sample_slice = self.vhr_dataset.isel(time=idx)

        era_data_arrays = [era_sample_slice[var_name].values for var_name in self.era5_variables]
        vhr_data_arrays = [vhr_sample_slice[var_name].values for var_name in self.vhr_variables]

        era_stacked_data_np = np.stack(era_data_arrays, axis=0)
        vhr_stacked_data_np = np.stack(vhr_data_arrays, axis=0)

        era_sample_tensor = torch.from_numpy(era_stacked_data_np).float()
        vhr_sample_tensor = torch.from_numpy(vhr_stacked_data_np).float()

        if self.transform:
            era_sample_tensor = self.transform(era_sample_tensor)
            vhr_sample_tensor = self.transform(vhr_sample_tensor)

        era_datetime64_val = era_sample_slice.valid_time.values
        vhr_datetime64_val = vhr_sample_slice.time.values
        
        era_time_numeric = pd.Timestamp(era_datetime64_val).timestamp()
        vhr_time_numeric = pd.Timestamp(vhr_datetime64_val).timestamp()

        # this is just for safety, it should never happen that the datasets get misaligned
        assert vhr_time_numeric == era_time_numeric, "TimeStamps don't match"

        return era_sample_tensor, vhr_sample_tensor
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), # bias=False because BN follows
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), # bias=False because BN follows
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels_up, in_channels_skip, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use a simple upsampling layer and then a conv to adjust channels
        # else, use a learnable transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After upsampling, in_channels_up remains. After cat with in_channels_skip, it's in_channels_up + in_channels_skip.
            self.conv = DoubleConv(in_channels_up + in_channels_skip, out_channels, mid_channels=in_channels_up + in_channels_skip // 2) # Example mid_channels
        else:
            # ConvTranspose2d typically halves the input channels (in_channels_up)
            self.up = nn.ConvTranspose2d(in_channels_up, in_channels_up // 2, kernel_size=2, stride=2)
            # After upsampling, channels are in_channels_up // 2. After cat with in_channels_skip, it's in_channels_up // 2 + in_channels_skip.
            self.conv = DoubleConv(in_channels_up // 2 + in_channels_skip, out_channels)


    def forward(self, x1_up, x2_skip):
        """
        x1_up: tensor from the upsampling path (lower layer in U-Net)
        x2_skip: tensor from the skip connection (encoder path)
        """
        x1_up = self.up(x1_up) # Upsample x1

        # Pad x1_up to match spatial dimensions of x2_skip if they differ
        # This can happen if input dimensions are odd.
        # Input is (N, C, H, W)
        diffY = x2_skip.size()[2] - x1_up.size()[2]
        diffX = x2_skip.size()[3] - x1_up.size()[3]

        # F.pad expects (padding_left, padding_right, padding_top, padding_bottom)
        x1_up = F.pad(x1_up, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2_skip, x1_up], dim=1) # Concatenate along channel dimension
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels_in, 64)     # x_skip1 (H, W)
        self.down1 = Down(64, 128)                   # x_skip2 (H/2, W/2)
        self.down2 = Down(128, 256)                  # x_skip3 (H/4, W/4)
        self.down3 = Down(256, 512)                  # x_skip4 (H/8, W/8)
        self.down4 = Down(512, 1024)                 # Bottleneck (H/16, W/16)

        # Decoder
        # Arguments for Up: (in_channels_up, in_channels_skip, out_channels, bilinear)
        # in_channels_up: channels from the layer below that is being upsampled.
        # in_channels_skip: channels from the corresponding skip connection.
        # out_channels: desired output channels for this Up block.

        self.up1 = Up(1024, 512, 512, bilinear) # Upsamples 1024 (from bottleneck), skips 512 (from down3)
        self.up2 = Up(512, 256, 256, bilinear)  # Upsamples 512 (from up1), skips 256 (from down2)
        self.up3 = Up(256, 128, 128, bilinear)  # Upsamples 256 (from up2), skips 128 (from down1)
        self.up4 = Up(128, 64, 64, bilinear)    # Upsamples 128 (from up3), skips 64 (from inc)
        
        self.outc = OutConv(64, n_channels_out)

    def forward(self, x):
        # Encoder
        x_skip1 = self.inc(x)
        x_skip2 = self.down1(x_skip1)
        x_skip3 = self.down2(x_skip2)
        x_skip4 = self.down3(x_skip3)
        bottleneck = self.down4(x_skip4)

        # Decoder
        x = self.up1(bottleneck, x_skip4)
        x = self.up2(x, x_skip3)
        x = self.up3(x, x_skip2)
        x = self.up4(x, x_skip1)
        
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    era5_path = "datasets/regridded_era5.nc"
    era5_data = xr.open_dataset(era5_path, mask_and_scale=True)
    print("--- ERA5 Data Loaded ---")
    # print(era5_data) # Keep it concise for now
    
    vhr_path = "datasets/vhr-rea.nc"
    vhr_data = xr.open_dataset(vhr_path, mask_and_scale=True)
    print("--- VHR Data Loaded ---")
    # print(vhr_data)

    # Example transforms (e.g., normalization - parameters should be pre-calculated)
    # from torchvision import transforms
    # input_norm = transforms.Normalize(mean=[mean_u10, mean_v10], std=[std_u10, std_v10]) # Example
    # target_norm = transforms.Normalize(mean=[mean_U, mean_V], std=[std_U, std_V])       # Example

    italy_weather_dataset = ItalyWeatherDataset(era5_data, vhr_data)

    print(f"\nDataset length: {len(italy_weather_dataset)}")

    batch_size = 2
    num_input_channels = 2
    num_output_channels = 2 
    height = 680
    width = 535

    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = UNet(n_channels_in=num_input_channels, n_channels_out=num_output_channels, bilinear=True).to(device)
    
    dataloader = DataLoader(italy_weather_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    for epoch in range(3):
        for era5_batch, vhr_batch in dataloader:
            era5_batch, vhr_batch = era5_batch.to(device), vhr_batch.to(device)
    
            optimizer.zero_grad()
            outputs = model(era5_batch)
            loss = criterion(outputs, vhr_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
