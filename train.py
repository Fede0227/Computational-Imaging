import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.nn import MSELoss
from tqdm import tqdm
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim_skimage # Keep for metric calculation
from model import ResUnet

# --- PyTorch SSIM Implementation ---
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



DATASET_PATH = "datasets/normalized_minmax/"
WEIGHTS_PATH = "models/unet_sr"
EPOCHS = 120
BATCH_SIZE = 8
EARLY_STOPPING_PATIENCE = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"\n--- Loading preprocessed data from {DATASET_PATH} ---")

train_data = torch.load(DATASET_PATH + "normalized_train_data.pt", map_location=device) # map_location can be helpful TODO: why?
train_dataset = TensorDataset(train_data["era5"], train_data["vhr"])
print(f"Loaded TensorDataset train with {len(train_dataset)} samples.")
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

validation_data = torch.load(DATASET_PATH + "normalized_val_data.pt", map_location=device)
validation_dataset = TensorDataset(validation_data["era5"], validation_data["vhr"])
print(f"Loaded TensorDataset validation with {len(validation_dataset)} samples. (WARN: Using test data for validation)")
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

test_data = torch.load(DATASET_PATH + "normalized_test_data.pt", map_location=device)
test_dataset = TensorDataset(test_data["era5"], test_data["vhr"])
print(f"Loaded TensorDataset test with {len(test_dataset)} samples.")
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = ResUnet(in_channels=2, out_channels=2)
model.to(device)

# --- CHOOSE AND DEFINE YOUR LOSS FUNCTION ---
LOSS_CHOICE = "L1_SSIM" # Options: "MSE", "SSIM", "L1_SSIM"
# LOSS_CHOICE = "SSIM"
# LOSS_CHOICE = "MSE"

# The `data_range` for SSIM should match your data normalization.
# If your data is normalized to [0, 1], data_range=1.0.
# If your data is normalized to [-1, 1], data_range=2.0.
# Your variable `DATASET_PATH = "datasets/normalized/"` suggests it's likely [0,1].
SSIM_DATA_RANGE = 1.0
NUM_CHANNELS = train_data["era5"].shape[1] # Get number of channels from data

if LOSS_CHOICE == "MSE":
    criterion = MSELoss()
    print("Using MSE Loss")
elif LOSS_CHOICE == "SSIM":
    criterion = SSIMLoss(data_range=SSIM_DATA_RANGE, channel=NUM_CHANNELS, size_average=True)
    print(f"Using SSIM Loss (data_range={SSIM_DATA_RANGE}, channels={NUM_CHANNELS})")
elif LOSS_CHOICE == "L1_SSIM":
    # Alpha is the weight for (1-SSIM) term, (1-alpha) for L1.
    # alpha=0.5 gives equal weight. alpha=0.84 is a common value from some research.
    criterion = L1SSIMLoss(alpha=0.84, ssim_data_range=SSIM_DATA_RANGE, ssim_channel=NUM_CHANNELS)
    print(f"Using L1+SSIM Loss (alpha=0.84 for SSIM part, data_range={SSIM_DATA_RANGE}, channels={NUM_CHANNELS})")
else:
    raise ValueError("Invalid LOSS_CHOICE")

criterion.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

print(f"Training UNet for {EPOCHS} with batch size of {BATCH_SIZE} using {LOSS_CHOICE} loss.")

best_val_loss = float('inf')
patience_counter = 0
best_model_state = None
early_stopped_epoch = -1

history = {
    'train_loss': torch.zeros((EPOCHS,)),
    'train_ssim': torch.zeros((EPOCHS,)),
    'val_loss': torch.zeros((EPOCHS,)),
    'val_ssim': torch.zeros((EPOCHS,))
}

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

epoch_pbar = tqdm(range(EPOCHS), desc="Training Epochs")
for epoch in epoch_pbar:
    model.train()
    train_loss_sum = 0
    train_ssim_sum = 0
    start_time = time.time()

    for t, (x_batch, y_batch) in enumerate(train_dataloader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(x_batch)
        
        loss = criterion(y_pred, y_batch) 
        
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        with torch.no_grad():
            train_ssim_sum += compute_ssim_for_batch(y_pred, y_batch, data_range_option=SSIM_DATA_RANGE)


    train_batch_count = len(train_dataloader)
    history['train_loss'][epoch] = train_loss_sum / train_batch_count
    history['train_ssim'][epoch] = train_ssim_sum / train_batch_count

    model.eval()
    val_loss_sum = 0.0
    val_ssim_sum = 0.0
    
    with torch.no_grad():
        for x_batch_val, y_batch_val in validation_dataloader:
            x_batch_val = x_batch_val.to(device)
            y_batch_val = y_batch_val.to(device)
            
            y_pred_val = model(x_batch_val)
            
            val_batch_loss = criterion(y_pred_val, y_batch_val)
            
            val_loss_sum += val_batch_loss.item()
            val_ssim_sum += compute_ssim_for_batch(y_pred_val, y_batch_val, data_range_option=SSIM_DATA_RANGE)
    
    validation_batch_count = len(validation_dataloader)
    history['val_loss'][epoch] = val_loss_sum / validation_batch_count
    history['val_ssim'][epoch] = val_ssim_sum / validation_batch_count
    
    elapsed_time_epoch = time.time() - start_time
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} ({elapsed_time_epoch:.1f}s) - "
              f"Loss: {history['train_loss'][epoch]:.4f}/{history['val_loss'][epoch]:.4f} - "
              f"SSIM (metric): {history['train_ssim'][epoch]:.4f}/{history['val_ssim'][epoch]:.4f}")

    current_val_loss = history["val_loss"][epoch]
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        if WEIGHTS_PATH is not None:
            torch.save(model.state_dict(), f"{WEIGHTS_PATH}_best_{LOSS_CHOICE}.pt")
            print(f"Saved new best model to {WEIGHTS_PATH}_best_{LOSS_CHOICE}.pt")
    else:
        patience_counter += 1
        if EARLY_STOPPING_PATIENCE is not None and patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            early_stopped_epoch = epoch + 1
            if best_model_state:
                 model.load_state_dict(best_model_state)
                 print("Restored best model weights.")
            break
epoch_pbar.close()

if WEIGHTS_PATH is not None:
    if early_stopped_epoch != -1:
        print(f"Training stopped early at epoch {early_stopped_epoch}. Best model already saved.")
    elif EPOCHS > 0 :
        torch.save(model.state_dict(), f"{WEIGHTS_PATH}_final_epoch_{EPOCHS}_{LOSS_CHOICE}.pt") # Add loss type
        print(f"Saved final model at epoch {EPOCHS} to {WEIGHTS_PATH}_final_epoch_{EPOCHS}_{LOSS_CHOICE}.pt")

if early_stopped_epoch != -1 and early_stopped_epoch < EPOCHS:
    for key in history:
        history[key] = history[key][:early_stopped_epoch]

print("\n--- Training Summary ---")
if len(history['train_loss']) > 0: # Check if any epochs ran
    print(f"Final train loss: {history['train_loss'][-1]:.4f}, SSIM (metric): {history['train_ssim'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}, SSIM (metric): {history['val_ssim'][-1]:.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")
else:
    print("No training epochs were completed.")
print("-" * 50)

model.eval()
test_loss_sum = 0
test_ssim_sum = 0

print(f"Evaluating the Unet using {LOSS_CHOICE} for loss calculation during test, and skimage for SSIM metric.")
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