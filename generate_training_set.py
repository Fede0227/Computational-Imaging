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

        era_metadata = {
            'valid_time': era_time_numeric,
        }

        vhr_metadata = {
            'time': vhr_time_numeric,
        }

        return era_sample_tensor, era_metadata, vhr_sample_tensor, vhr_metadata
    

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

    if len(italy_weather_dataset) > 0:
        sample_idx = 0
        era5_sample, era5_sample_metadata, vhr_sample, vhr_sample_metadata  = italy_weather_dataset[sample_idx]
        print(f"\n--- Sample {sample_idx} ---")
        print(f"--- Low-Res Input (ERA5) ---")
        print(f"Tensor shape: {era5_sample.shape}") 
        print(f"Tensor dtype: {era5_sample.dtype}")
        print(f"--- High-Res Target (VHR) ---")
        print(f"Tensor shape: {vhr_sample.shape}") 
        print(f"Tensor dtype: {vhr_sample.dtype}")
        print(f"-----------------------------")

        plot_dataset_sample(era5_sample, vhr_sample) 

        print("\n--- Testing with DataLoader ---")
        dataloader = DataLoader(italy_weather_dataset, batch_size=4, shuffle=True, num_workers=0)

        era5_batch, era5_batch_metadata, vhr_batch, vhr_batch_metadata = next(iter(dataloader))
        
        print("\n--- Batch from DataLoader ---")
        print(f"Batch Low-Res Input shape: {era5_batch.shape}")
        print(f"Batch High-Res Target shape: {vhr_batch.shape}")
        print(f"Batch Low-Res dtype: {era5_batch.dtype}")
        print(f"Batch High-Res dtype: {vhr_batch.dtype}")
    else:
        print("Dataset is empty, cannot test DataLoader.")