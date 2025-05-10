import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # For robust timestamp handling

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


class SuperResolutionWeatherDataset(Dataset):
    era5_vars = ["u10", "v10"]
    vhr_vars = ["U_10M", "V_10M"]
    
    def __init__(self, era5_xarray_dataset, vhr_xarray_dataset, input_transform=None, target_transform=None):
        super().__init__()
        self.era5_ds = era5_xarray_dataset
        self.vhr_ds = vhr_xarray_dataset
        self.input_transform = input_transform
        self.target_transform = target_transform

        # --- Robust Timestamp Alignment ---
        era5_times = pd.to_datetime(self.era5_ds.valid_time.values)
        vhr_times = pd.to_datetime(self.vhr_ds.time.values)

        # Find common timestamps
        # Using sets for efficient intersection
        common_timestamps_set = set(era5_times).intersection(set(vhr_times))
        if not common_timestamps_set:
            raise ValueError("No common timestamps found between ERA5 and VHR datasets. "
                             "Cannot create paired samples for super-resolution.")
        
        # Sort them to ensure consistent ordering
        self.common_timestamps = sorted(list(common_timestamps_set))
        
        # Create a mapping from our new index (0 to len(common_timestamps)-1)
        # to the original indices in era5_ds and vhr_ds
        self.era5_time_to_idx_map = {time: i for i, time in enumerate(era5_times)}
        self.vhr_time_to_idx_map = {time: i for i, time in enumerate(vhr_times)}
        
        self.paired_indices = []
        for common_t in self.common_timestamps:
            if common_t in self.era5_time_to_idx_map and common_t in self.vhr_time_to_idx_map:
                self.paired_indices.append({
                    "common_time": common_t,
                    "era5_idx": self.era5_time_to_idx_map[common_t],
                    "vhr_idx": self.vhr_time_to_idx_map[common_t]
                })
            else:
                # This should not happen if common_t came from intersection, but as a safeguard:
                print(f"Warning: Timestamp {common_t} was in intersection but not found in map. Skipping.")

        self.num_samples = len(self.paired_indices)
        if self.num_samples == 0:
             raise ValueError("No valid paired samples could be constructed after aligning timestamps.")
        print(f"Initialized dataset with {self.num_samples} paired samples.")


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not 0 <= idx < self.num_samples:
            raise IndexError(f"Index {idx} is out of bounds for dataset with size {self.num_samples}")

        pair_info = self.paired_indices[idx]
        era5_original_idx = pair_info["era5_idx"]
        vhr_original_idx = pair_info["vhr_idx"]

        # Select data for the given time index using .isel for integer-based selection
        era_sample_slice = self.era5_ds.isel(valid_time=era5_original_idx)
        vhr_sample_slice = self.vhr_ds.isel(time=vhr_original_idx)

        # Extract and stack low-resolution (input) data
        era_data_arrays = [era_sample_slice[var_name].data for var_name in self.era5_vars] # .data is often preferred over .values
        low_res_input_np = np.stack(era_data_arrays, axis=0)
        low_res_input_tensor = torch.from_numpy(low_res_input_np.astype(np.float32)) # Ensure float32

        # Extract and stack high-resolution (target) data
        vhr_data_arrays = [vhr_sample_slice[var_name].data for var_name in self.vhr_vars]
        high_res_target_np = np.stack(vhr_data_arrays, axis=0)
        high_res_target_tensor = torch.from_numpy(high_res_target_np.astype(np.float32))

        if self.input_transform:
            low_res_input_tensor = self.input_transform(low_res_input_tensor)
        
        if self.target_transform:
            high_res_target_tensor = self.target_transform(high_res_target_tensor)
            
        # return low_res_input_tensor, high_res_target_tensor, pd.Timestamp(pair_info["common_time"]).timestamp()
        return low_res_input_tensor, high_res_target_tensor

# --- Main execution part ---
if __name__ == "__main__":
    era5_path = "datasets/regridded_era5.nc"
    era5_data = xr.open_dataset(era5_path, mask_and_scale=True)
    print("--- ERA5 Data Loaded ---")
    # print(era5_data) # Keep it concise for now
    
    vhr_path = "datasets/vhr-rea.nc"
    vhr_data = xr.open_dataset(vhr_path, mask_and_scale=True)
    print("--- VHR Data Loaded ---")
    # print(vhr_data)

    # TODO: NORMALIZATION?

    italy_weather_dataset = SuperResolutionWeatherDataset(era5_data, vhr_data)

    print(f"\nDataset length: {len(italy_weather_dataset)}")

    if len(italy_weather_dataset) > 0:
        sample_idx = 0
        low_res_sample, high_res_sample = italy_weather_dataset[sample_idx]
        print(f"\n--- Sample {sample_idx} ---")
        print(f"--- Low-Res Input (ERA5) ---")
        print(f"Tensor shape: {low_res_sample.shape}") 
        print(f"Tensor dtype: {low_res_sample.dtype}")
        print(f"--- High-Res Target (VHR) ---")
        print(f"Tensor shape: {high_res_sample.shape}") 
        print(f"Tensor dtype: {high_res_sample.dtype}")
        print(f"-----------------------------")

        # Your plot_dataset_sample would need to be adapted if you use it,
        # as it expects the old metadata structure.
        # plot_dataset_sample(low_res_sample, high_res_sample) 

        print("\n--- Testing with DataLoader ---")
        # Note: If your transforms include random augmentations, shuffle=True is good.
        # num_workers > 0 can speed up data loading if transforms are CPU-intensive.
        dataloader = DataLoader(italy_weather_dataset, batch_size=4, shuffle=True, num_workers=0)

        # Get one batch
        batch_low_res, batch_high_res = next(iter(dataloader))
        
        print("\n--- Batch from DataLoader ---")
        print(f"Batch Low-Res Input shape: {batch_low_res.shape}")   # Expected: (batch_size, 2, rlat, rlon)
        print(f"Batch High-Res Target shape: {batch_high_res.shape}") # Expected: (batch_size, 2, rlat, rlon)
        print(f"Batch Low-Res dtype: {batch_low_res.dtype}")
    else:
        print("Dataset is empty, cannot test DataLoader.")