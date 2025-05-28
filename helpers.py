import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

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
    plt.title('ERA5 Vector Field Intensity Map')
    plt.xlabel('Pixel X-coordinate (width)')
    plt.ylabel('Pixel Y-coordinate (height)')
    plt.colorbar(label='Wind Magnitude')
 
    plt.subplot(1, 2, 2)
    plt.imshow(vhr_magnitude_numpy, cmap='viridis', origin='lower')
    plt.title('VHR Vector Field Intensity Map')
    plt.xlabel('Pixel X-coordinate (width)')
    plt.ylabel('Pixel Y-coordinate (height)')
    plt.colorbar(label='Wind Magnitude')
 
    plt.tight_layout()
    plt.show()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, data_range, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1) # Mean per image in batch

class SSIM(nn.Module):
    def __init__(self, window_size=11, data_range=1.0, size_average=True, channel=1):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.data_range = data_range # Important: Set based on your data normalization
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
        
        return _ssim(img1, img2, window, self.window_size, channel, self.data_range, self.size_average)

class SSIMLoss(SSIM):
    """
    SSIM Loss. Higher SSIM is better, so loss is 1 - SSIM.
    Assumes inputs are in range [0, data_range] or [-data_range/2, data_range/2] etc.
    For normalized data [0,1], data_range=1.0. For [-1,1], data_range=2.0.
    """
    def __init__(self, window_size=11, data_range=1.0, size_average=True, channel=1):
        super(SSIMLoss, self).__init__(window_size, data_range, size_average, channel)
        self.channel = channel # Ensure channel is passed for window creation

    def forward(self, img1, img2):
        # SSIM is a similarity metric (0 to 1, higher is better)
        # For loss, we want to minimize (1 - SSIM)
        return 1.0 - super(SSIMLoss, self).forward(img1, img2)

class L1SSIMLoss(nn.Module):
    def __init__(self, alpha=0.84, ssim_window_size=11, ssim_data_range=1.0, ssim_channel=1):
        """
        Combined L1 and SSIM Loss.
        Loss = alpha * (1-SSIM) + (1-alpha) * L1
        alpha: weight for SSIM loss. Common values are 0.5 or 0.84 (from some papers).
        """
        super(L1SSIMLoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss() # Mean Absolute Error
        self.ssim_loss_fn = SSIMLoss(window_size=ssim_window_size, data_range=ssim_data_range, channel=ssim_channel)

    def forward(self, y_pred, y_true):
        ssim_val_loss = self.ssim_loss_fn(y_pred, y_true)
        l1_val_loss = self.l1_loss(y_pred, y_true)
        
        combined_loss = self.alpha * ssim_val_loss + (1 - self.alpha) * l1_val_loss
        return combined_loss
