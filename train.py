import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import MSELoss
from tqdm import tqdm
import time
from model import ResidualUNet
from helpers import L1SSIMLoss, MSESSIMLoss, compute_ssim_for_batch
import numpy as np
import random

COORDINATES = os.getenv(key="COORDINATES", default=0)

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
print(f"Using device: {device}")

DATASET_PATH = "datasets/normalized_minmax_vectors/"
COORDINATES_TYPE = "vectors"
if COORDINATES:
    DATASET_PATH = "datasets/normalized_minmax_direction/"
    COORDINATES_TYPE = "direction"

MODEL_PATH = "models/"
EPOCHS = 200
BATCH_SIZE = 4
NUM_CHANNELS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
SSIM_DATA_RANGE = 1.0

LOSS = "mse"
if LOSS == "mse": criterion = MSELoss()
if LOSS == "msessim": criterion = MSESSIMLoss()
if LOSS == "l1ssim": criterion = L1SSIMLoss()
criterion.to(device)

MODEL_NAME = f"unet_{COORDINATES_TYPE}_{LOSS}_loss_{EPOCHS}_epochs_{BATCH_SIZE}_batch_1em3_lr_1em5_weightdecay"

print(f"\n--- Loading preprocessed data from {DATASET_PATH} ---")

train_data = torch.load(DATASET_PATH + "normalized_train_data.pt", map_location=device)
train_dataset = TensorDataset(train_data["era5"], train_data["vhr"])
print(f"Loaded TensorDataset train with {len(train_dataset)} samples.")
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

test_data = torch.load(DATASET_PATH + "normalized_test_data.pt", map_location=device)
test_dataset = TensorDataset(test_data["era5"], test_data["vhr"])
print(f"Loaded TensorDataset test with {len(test_dataset)} samples.")
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = ResidualUNet(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS)
model.to(device)

print("-"*50)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
print(f"Training with precision: {next(model.parameters()).dtype}")
print("-"*50)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

print(f"Training UNet for {EPOCHS} with batch size {BATCH_SIZE}, {LOSS} loss, {LEARNING_RATE} learning rate.")

best_val_loss = float("inf")
best_model_state = None
best_model_epoch = -1
history = {
    "train_loss": torch.zeros((EPOCHS,)),
    "train_ssim": torch.zeros((EPOCHS,)),
    "val_loss": torch.zeros((EPOCHS,)),
    "val_ssim": torch.zeros((EPOCHS,))
}

epoch_pbar = tqdm(range(EPOCHS), desc="Training Epochs")
for epoch in epoch_pbar:
    model.train()
    train_loss_sum = 0
    train_ssim_sum = 0
    start_time = time.time()

    for x_batch, y_batch in train_dataloader:
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
    history["train_loss"][epoch] = train_loss_sum / train_batch_count
    history["train_ssim"][epoch] = train_ssim_sum / train_batch_count

    model.eval()
    val_loss_sum = 0.0
    val_ssim_sum = 0.0

    with torch.no_grad():
        for x_batch_val, y_batch_val in test_dataloader:
            x_batch_val = x_batch_val.to(device)
            y_batch_val = y_batch_val.to(device)

            y_pred_val = model(x_batch_val)

            val_batch_loss = criterion(y_pred_val, y_batch_val)
            
            val_loss_sum += val_batch_loss.item()
            val_ssim_sum += compute_ssim_for_batch(y_pred_val, y_batch_val, data_range_option=SSIM_DATA_RANGE)
    
    validation_batch_count = len(test_dataloader)
    history["val_loss"][epoch] = val_loss_sum / validation_batch_count
    history["val_ssim"][epoch] = val_ssim_sum / validation_batch_count
    
    elapsed_time_epoch = time.time() - start_time
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
            f"Train: Loss: {history['train_loss'][epoch]:.6f} - SSIM: {history['train_ssim'][epoch]:.6f} "
            f"Val:   Loss: {history['val_loss'][epoch]:.6f} - SSIM: {history['val_ssim'][epoch]:.6f}")

    current_val_loss = history["val_loss"][epoch]
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        best_model_state = model.state_dict().copy()
        best_model_epoch = epoch
        if MODEL_PATH is not None:
            torch.save(model.state_dict(), f"{MODEL_PATH + MODEL_NAME}_best.pt")
            print(f"Saved new best model to {MODEL_PATH + MODEL_NAME}_best.pt")

epoch_pbar.close()

if MODEL_PATH is not None:
    torch.save(model.state_dict(), f"{MODEL_PATH + MODEL_NAME}_final.pt")
    print(f"Saved final model to {MODEL_PATH + MODEL_NAME}_final.pt")

if best_val_loss != history["val_loss"][-1]:
    model.load_state_dict(best_model_state)
    print(f"Reloaded the best model since it wasnt the one at last epoch")

print("\n----- Training Summary -----")
if len(history["train_loss"]) > 0:
    print(f"Final train loss: {history['train_loss'][-1]:.6f}, SSIM: {history['train_ssim'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}, SSIM: {history['val_ssim'][-1]:.6f}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"The best model was found at epoch {best_model_epoch}")
    print("\n----- Full Training History -----")
    print(history)
else:
    print("No training epochs were completed.")
print("-" * 50)

model.eval()
test_loss_sum = 0
test_ssim_sum = 0

print(f"Evaluating the Unet using {LOSS} loss")
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
    print(f"Test loss: {avg_test_loss:.6f}, Test SSIM: {avg_test_ssim:.6f}")
else:
    print("Test dataloader is empty.")
