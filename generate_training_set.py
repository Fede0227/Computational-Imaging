import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm

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

    def extract_region(self, data_tensor):
        patch_h, patch_w = (256, 256)
        _, h, w = data_tensor.shape
        
        # Take the center patch
        x = (w - patch_w) // 2
        y = (h - patch_h) // 2
        
        # Extract the region
        region = data_tensor[:, y:y+patch_h, x:x+patch_w]
        return region

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

        # dataset cropping
        era_sample_tensor = self.extract_region(era_sample_tensor)
        vhr_sample_tensor = self.extract_region(vhr_sample_tensor)
    
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


if __name__ == "__main__":
    era5_path = "datasets/regridded_era5.nc"
    era5_data = xr.open_dataset(era5_path, mask_and_scale=True)
    print("--- ERA5 Data Loaded ---")
    print(era5_data) # Keep it concise for now
    
    vhr_path = "datasets/vhr-rea.nc"
    vhr_data = xr.open_dataset(vhr_path, mask_and_scale=True)
    print("--- VHR Data Loaded ---")
    print(vhr_data)

    italy_weather_dataset = ItalyWeatherDataset(era5_data, vhr_data)
    print(f"\nFull dataset length: {len(italy_weather_dataset)}")

    era_tensors_list = []
    vhr_tensors_list = []

    print(f"\n--- Extracting and collecting all {len(italy_weather_dataset)} samples ---")
    for i in tqdm(range(len(italy_weather_dataset)), desc="Processing samples"):
        era_sample, vhr_sample = italy_weather_dataset[i]
        era_tensors_list.append(era_sample)
        vhr_tensors_list.append(vhr_sample)

    print("\n--- Stacking tensors ---")
    all_era_samples_stacked = torch.stack(era_tensors_list)
    all_vhr_samples_stacked = torch.stack(vhr_tensors_list)

    print(f"Shape of stacked ERA5 tensors: {all_era_samples_stacked.shape}")
    print(f"Shape of stacked VHR tensors: {all_vhr_samples_stacked.shape}")

    output_path = "datasets/preprocessed_italy_weather.pt"

    print(f"\n--- Saving preprocessed tensors to {output_path} ---")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'era5': all_era_samples_stacked,
        'vhr': all_vhr_samples_stacked
    }, output_path)
    print("Successfully saved preprocessed data.")
