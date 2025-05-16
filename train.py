import os
import time

from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset
from skimage.metrics import structural_similarity as ssim
import numpy as np
from model import ResUnet

device = "cpu"
if torch.cuda.is_available:
    print('GPU available')
    device = torch.device("cuda:0")
if torch.mps.is_available:
    print("USING MPs")
    device = torch.device("mps")
else:
    print('Training on CPU')

print(f"USING DEVICE {device}")

DATASET_PATH = "datasets/normalized/"
EPOCHS = 50
BATCH_SIZE = 2
EARLY_STOPPING_PATIENCE = 5

print(f"\n--- Loading preprocessed data from {DATASET_PATH} ---")

train_data = torch.load(DATASET_PATH + "normalized_train_data.pt")
train_dataset = TensorDataset(train_data["era5"], train_data["vhr"])
print(f"Loaded TensorDataset train with {len(train_dataset)} samples.")
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

validation_data = torch.load(DATASET_PATH + "normalized_test_data.pt")
validation_dataset = TensorDataset(validation_data["era5"], validation_data["vhr"])
print(f"Loaded TensorDataset test with {len(validation_dataset)} samples.")
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

test_data = torch.load(DATASET_PATH + "normalized_test_data.pt")
test_dataset = TensorDataset(test_data["era5"], test_data["vhr"])
print(f"Loaded TensorDataset test with {len(test_dataset)} samples.")
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = ResUnet(in_channels=2, out_channels=2)
model.to(device)

mse_loss = MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

print(f"Training UNet for {EPOCHS} with batch size of {BATCH_SIZE}")

# early stopping
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

history = {
    'train_loss': torch.zeros((EPOCHS,)),
    'train_ssim': torch.zeros((EPOCHS,)),
    'val_loss': torch.zeros((EPOCHS,)),
    'val_ssim': torch.zeros((EPOCHS,))
}

weights_path = "models/unet_sr"



def compute_ssim_for_batch(y_pred_tensor, y_true_tensor, data_range_option="dynamic_channel_wise"):
    """
    Computes average SSIM for a batch of images.
    Assumes y_pred_tensor and y_true_tensor are (B, C, H, W) PyTorch tensors.
    data_range_option:
        "dynamic_channel_wise": data_range is true_channel.max() - true_channel.min()
        "fixed_0_1": data_range is 1.0 (assumes data normalized to [0,1])
        Any other float value will be used as a fixed data_range.
    """
    y_pred_np = y_pred_tensor.detach().cpu().numpy()
    y_true_np = y_true_tensor.detach().cpu().numpy()
    
    batch_size = y_true_np.shape[0]
    num_channels = y_true_np.shape[1]
    
    total_ssim_for_batch = 0.0
    
    for i in range(batch_size): # Iterate over images in batch
        item_ssim_sum_channels = 0.0
        for ch_idx in range(num_channels): # Iterate over channels
            pred_ch = y_pred_np[i, ch_idx]
            true_ch = y_true_np[i, ch_idx]
            
            current_data_range = None
            if data_range_option == "dynamic_channel_wise":
                current_data_range = true_ch.max() - true_ch.min()
                if current_data_range == 0: # Handle constant image channel
                    if np.array_equal(pred_ch, true_ch):
                        item_ssim_sum_channels += 1.0
                    else:
                        item_ssim_sum_channels += 0.0 
                    continue 
            elif data_range_option == "fixed_0_1":
                current_data_range = 1.0
            elif isinstance(data_range_option, float):
                current_data_range = data_range_option
            
            # Ensure win_size is appropriate for image dimensions if they are small
            # Default win_size=7. If H or W < 7, ssim can error.
            # Consider making win_size dynamic: min_dim = min(pred_ch.shape); ws = min(7, min_dim) if min_dim % 2 == 1 else min(7, min_dim -1); ws = max(3, ws)
            # For simplicity, assuming images are large enough for default win_size.
            try:
                item_ssim_sum_channels += ssim(pred_ch, true_ch, data_range=current_data_range)
            except ValueError as e:
                # e.g., "Input array dimensions must be large enough for the chosen `win_size`."
                print(f"Warning: SSIM calculation error for item {i} chan {ch_idx}. Shape: {true_ch.shape}. Error: {e}. Using SSIM=0.")
                item_ssim_sum_channels += 0.0 # Or handle differently

        total_ssim_for_batch += item_ssim_sum_channels / num_channels # Average SSIM over channels for this item
        
    return total_ssim_for_batch / batch_size # Average SSIM for the batch


for epoch in tqdm(range(EPOCHS), desc="Training Epochs"):
    model.train()
    train_loss = 0
    train_ssim = 0
    start_time = time.time()
    
    for t, (x, y) in enumerate(train_dataloader):
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()

        y_pred = model(x)
        loss = mse_loss(y_pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        with torch.no_grad():
        #    train_ssim += ssim(y_pred, y)
            train_ssim += compute_ssim_for_batch(y_pred, y, data_range_option="fixed_0_1") # NEW (assuming normalized [0,1])
        
        # Measure time and show progress
        if (t + 1) % 5 == 0 or t == len(train_dataloader) - 1:  # Update every 5 batches to reduce output spam
            elapsed_time = time.time() - start_time
            print(
                f"({elapsed_time}) Epoch ({epoch+1}/{EPOCHS}) Batch {t+1}/{len(train_dataloader)} -> "
                f"Train Loss = {train_loss/(t+1):.4f}, Train SSIM = {train_ssim/(t+1):.4f}",
                end="\r",
            )

        # Average training metrics
        train_batch_count = len(train_dataloader)
        history['train_loss'][epoch] = train_loss / train_batch_count
        history['train_ssim'][epoch] = train_ssim / train_batch_count

        # ---------- Validation Phase ----------
        model.eval()
        val_loss = 0.0
        val_ssim = 0.0
        
        with torch.no_grad():
            for x, y in validation_dataloader:
                x = x.to(device)
                y = y.to(device)
                
                # Forward pass only (no backprop in validation)
                y_pred = model(x)
                loss = mse_loss(y_pred, y)
                
                # Update metrics
                val_loss += loss.item()
                # val_ssim += ssim(y_pred, y)
                val_ssim += compute_ssim_for_batch(y_pred, y, data_range_option="fixed_0_1") # NEW (assuming normalized [0,1])
        
        # Average validation metrics
        validation_batch_count = len(validation_dataloader)
        history['val_loss'][epoch] = val_loss / validation_batch_count
        history['val_ssim'][epoch] = val_ssim / validation_batch_count
        
        # Print epoch results
        print(
            f"\nEpoch {epoch+1}/{EPOCHS} - "
            f"Train Loss: {history['train_loss'][epoch]:.4f}, Train SSIM: {history['train_ssim'][epoch]:.4f}, "
            f"Val Loss: {history['val_loss'][epoch]:.4f}, Val SSIM: {history['val_ssim'][epoch]:.4f}"
        )

        # early stopping check
        current_val_loss = history["val_loss"][epoch]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            # Save best model
            if weights_path is not None:
                torch.save(model.state_dict(), f"{weights_path}_best.pt")
        else:
            patience_counter += 1
            if EARLY_STOPPING_PATIENCE is not None and patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                # Restore best model and break to test
                model.load_state_dict(best_model_state)
                break

# save model if not done through early stopping
if weights_path is not None and (EARLY_STOPPING_PATIENCE is None or patience_counter < EARLY_STOPPING_PATIENCE):
    torch.save(model.state_dict(), f"{weights_path}_final.pt")

print(history)

# test the model 
model.eval()

test_loss = 0
test_ssim = 0

print("Evaluating the Unet")
start_time = time.time()

with torch.no_grad():
    for x, y in test_dataloader:
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)

        loss = mse_loss(y_pred, y)
        test_loss += loss.item()
        test_ssim += compute_ssim_for_batch(y_pred, y, data_range_option="fixed_0_1")

num_batches = len(test_dataloader)
avg_loss = test_loss / num_batches
avg_ssim = test_ssim / num_batches

print(f"Test completed in {time.time() - start_time}")
print(f"Test loss: {avg_loss}, test SSIM: {avg_ssim}")

# print results
model.eval()
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