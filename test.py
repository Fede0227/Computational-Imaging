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
from helpers import L1SSIMLoss, compute_ssim_for_batch
import random

def fix_random(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

fix_random(seed=1337)

device = "cpu"
if torch.cuda.is_available(): device = torch.device("cuda:0")
if torch.mps.is_available(): device = torch.device("mps")

DATASET_PATH = "datasets/normalized_minmax_vectors/"
BATCH_SIZE = 4
NUM_SAMPLES = 3
NUM_CHANNELS = 2
SSIM_DATA_RANGE = 1.0
MODEL_PATH = "models/FINAL_unet_vectors_mse_loss_200_epochs_4_batch_1em3_lr_1em5_weightdecay_best.pt"
# MODEL_PATH = "models/FINAL_unet_vectors_l1ssim_loss_200_epochs_4_batch_1em3_lr_1em5_weightdecay_best.pt"

test_data = torch.load(DATASET_PATH + "normalized_test_data.pt")
test_dataset = TensorDataset(test_data["era5"], test_data["vhr"])
print(f"Loaded TensorDataset test with {len(test_dataset)} samples.")
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

criterion = MSELoss()
# criterion = L1SSIMLoss()
criterion.to(device)

model = ResidualUNet(in_channels=2, out_channels=2)
model.to(device)

state_dict = torch.load(MODEL_PATH, map_location=torch.device('mps'))
model.load_state_dict(state_dict)
print("Model weights loaded successfully.")

model.eval()
test_loss_sum = 0
test_ssim_sum = 0

print(f"Evaluating the Unet")
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
    print(f"Test loss: {avg_test_loss:.5f}, Test SSIM (metric): {avg_test_ssim:.5f}")
else:
    print("Test dataloader is empty.")



dataiter = iter(test_dataloader)
x_batch, y_batch = next(dataiter)


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
plt.savefig(MODEL_PATH + ".png")
plt.show()
