import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import MSELoss
import time
from model import ResidualUNet
from skimage.metrics import structural_similarity as ssim_skimage
from helpers import L1SSIMLoss

def compute_ssim_for_batch(y_pred_tensor, y_true_tensor, data_range_option="dynamic_channel_wise"):
    """
    Computes average SSIM for a batch of images using skimage.metrics.ssim.
    Assumes y_pred_tensor and y_true_tensor are (B, C, H, W) PyTorch tensors.
    data_range_option:
        "dynamic_channel_wise": data_range is true_channel.max() - true_channel.min()
        "fixed_0_1": data_range is 1.0 (assumes data normalized to [0,1])
        Any other float value will be used as a fixed data_range.
    """
    y_pred_np = y_pred_tensor.detach().cpu().numpy()
    y_true_np = y_true_tensor.detach().cpu().numpy()
    
    batch_size = y_true_np.shape[0]
    num_channels_data = y_true_np.shape[1]
    
    total_ssim_for_batch = 0.0
    
    for i in range(batch_size): 
        item_ssim_sum_channels = 0.0
        for ch_idx in range(num_channels_data): 
            pred_ch = y_pred_np[i, ch_idx]
            true_ch = y_true_np[i, ch_idx]
            
            current_data_range = None
            if data_range_option == "dynamic_channel_wise":
                min_val, max_val = true_ch.min(), true_ch.max()
                current_data_range = max_val - min_val
                if current_data_range < 1e-6: # Handle constant image channel or very small range
                    if np.allclose(pred_ch, true_ch): # Use allclose for float comparison
                        item_ssim_sum_channels += 1.0
                    else:
                        item_ssim_sum_channels += 0.0 
                    continue 
            elif data_range_option == "fixed_0_1":
                current_data_range = 1.0
            elif isinstance(data_range_option, float):
                current_data_range = data_range_option
            else: # Fallback if not recognized
                current_data_range = true_ch.max() - true_ch.min()
                if current_data_range < 1e-6:
                    current_data_range = 1.0 # Default if flat, to avoid div by zero in ssim if pred/true differ

            try:
                # skimage ssim expects channel axis last for multichannel, but here we do it per channel
                ssim_val = ssim_skimage(pred_ch, true_ch, data_range=current_data_range, channel_axis=None, win_size=min(7, min(pred_ch.shape)-1 if min(pred_ch.shape)%2==0 else min(pred_ch.shape)))
                item_ssim_sum_channels += ssim_val
            except ValueError as e:
                # print(f"Warning: SSIM calculation error for item {i} chan {ch_idx}. Shape: {true_ch.shape}. Range: {current_data_range}. Error: {e}. Using SSIM=0.")
                if np.allclose(pred_ch, true_ch): item_ssim_sum_channels += 1.0
                else: item_ssim_sum_channels += 0.0 

        total_ssim_for_batch += item_ssim_sum_channels / num_channels_data 
        
    return total_ssim_for_batch / batch_size



device = "cpu"
if torch.cuda.is_available: device = torch.device("cuda:0")
if torch.mps.is_available: device = torch.device("mps")

DATASET_PATH = "datasets/normalized_minmax/"
BATCH_SIZE = 8

test_data = torch.load(DATASET_PATH + "normalized_test_data.pt")
test_dataset = TensorDataset(test_data["era5"], test_data["vhr"])
print(f"Loaded TensorDataset test with {len(test_dataset)} samples.")
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

NUM_CHANNELS = 2
SSIM_DATA_RANGE = 1.0

# criterion = MSELoss()
criterion = L1SSIMLoss()
criterion.to(device)

model = ResidualUNet(in_channels=2, out_channels=2)
model.to(device)

# state_dict = torch.load("models/unet_sr_best_MSE.pt", map_location=torch.device('mps'))
state_dict = torch.load("models/unet_sr_final_epoch_100_l1ssim.pt", map_location=torch.device('mps'))
model.load_state_dict(state_dict)
print("Model weights loaded successfully.")

model.eval()
test_loss_sum = 0
test_ssim_sum = 0

print(f"Evaluating the Unet using MSE for loss calculation during test, and skimage for SSIM metric.")
start_time = time.time()

with torch.no_grad():
    for x_batch_test, y_batch_test in test_dataloader:
        x_batch_test = x_batch_test.to(device)
        y_batch_test = y_batch_test.to(device)

        y_pred_test = model(x_batch_test)

        test_batch_loss = criterion(y_pred_test, y_batch_test)
        test_loss_sum += test_batch_loss.item()
        test_ssim_sum += compute_ssim_for_batch(y_pred_test, y_batch_test, data_range_option=SSIM_DATA_RANGE)

num_batches_test = len(test_dataloader)
if num_batches_test > 0:
    avg_test_loss = test_loss_sum / num_batches_test
    avg_test_ssim = test_ssim_sum / num_batches_test
    print(f"Test completed in {time.time() - start_time:.2f}s")
    print(f"Test loss: {avg_test_loss:.4f}, Test SSIM (metric): {avg_test_ssim:.4f}")
else:
    print("Test dataloader is empty.")



dataiter = iter(test_dataloader)
x_batch, y_batch = next(dataiter)

NUM_SAMPLES = 3

x_samples = x_batch[:NUM_SAMPLES].to(device)
y_samples = y_batch[:NUM_SAMPLES].cpu()

with torch.no_grad():
    predictions = model(x_samples).cpu()

import matplotlib.pyplot as plt

# Plot results
fig, axes = plt.subplots(NUM_SAMPLES, 3, figsize=(15, 5*NUM_SAMPLES))

for i in range(NUM_SAMPLES):
    # Calculate magnitude for better visualization
    input_mag = torch.sqrt(x_batch[i, 0, :, :]**2 + x_batch[i, 1, :, :]**2).numpy()
    target_mag = torch.sqrt(y_samples[i, 0, :, :]**2 + y_samples[i, 1, :, :]**2).numpy()
    pred_mag = torch.sqrt(predictions[i, 0, :, :]**2 + predictions[i, 1, :, :]**2).numpy()
    
    # Plot input (low resolution)
    axes[i, 0].imshow(input_mag, cmap='viridis')
    axes[i, 0].set_title('Low Resolution Input')
    axes[i, 0].axis('off')
    
    # Plot target (high resolution)
    axes[i, 1].imshow(target_mag, cmap='viridis')
    axes[i, 1].set_title('High Resolution Target')
    axes[i, 1].axis('off')
    
    # Plot prediction
    axes[i, 2].imshow(pred_mag, cmap='viridis')
    axes[i, 2].set_title('Model Prediction')
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()