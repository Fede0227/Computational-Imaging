import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np

class ItalyWeatherDataset(Dataset):
    
    era5_variables = ["u10", "v10"]
    vhr_variables = ["U_10M", "V_10M"]
    
    def __init__(self, era5_dataset, vhr_dataset , transform=None, target_transform=None):
        super().__init__()
        self.era5_dataset = era5_dataset
        self.vhr_dataset = vhr_dataset
        self.transform = transform
        self.target_transform = target_transform # Placeholder for now

        self.num_samples = len(self.era5_dataset.valid_time) # TODO: assert that era5 and vhr have the same length


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not 0 <= idx < self.num_samples:
            raise IndexError(f"Index {idx} is out of bounds for dataset with size {self.num_samples}")

        # Select data for the given time index using .isel for integer-based selection
        era_sample_slice = self.era5_dataset.isel(valid_time=idx)
        vhr_sample_slice = self.vhr_dataset.isel(time=idx)

        # Extract the specified variables and stack them
        # .data is used to get the underlying numpy array (or dask array which then gets computed)
        # .values is an alias for .data that also computes dask arrays
        era_data_arrays = [era_sample_slice[var_name].values for var_name in self.era5_variables]
        vhr_data_arrays = [vhr_sample_slice[var_name].values for var_name in self.vhr_variables]

        # Stack along a new "channel" dimension (axis=0)
        # This results in a NumPy array of shape (num_variables, rlat_dim, rlon_dim)
        era_stacked_data_np = np.stack(era_data_arrays, axis=0)
        vhr_stacked_data_np = np.stack(vhr_data_arrays, axis=0)

        # Convert to PyTorch tensor
        era_sample_tensor = torch.from_numpy(era_stacked_data_np).float()
        vhr_sample_tensor = torch.from_numpy(vhr_stacked_data_np).float()

        if self.transform:
            era_sample_tensor = self.transform(era_sample_tensor)
            vhr_sample_tensor = self.transform(vhr_sample_tensor)

        era_metadata = {
            'valid_time': pd.to_datetime(era_sample_slice.valid_time.values),
            'expver': era_sample_slice.expver.values.item() if 'expver' in era_sample_slice else None,
            'number': era_sample_slice.number.values.item() if 'number' in era_sample_slice else None,
        }

        vhr_metadata = {
            'time': pd.to_datetime(vhr_sample_slice.time.values),
            'expver': vhr_sample_slice.expver.values.item() if 'expver' in vhr_sample_slice else None,
            'number': vhr_sample_slice.number.values.item() if 'number' in vhr_sample_slice else None,
        }

        return era_sample_tensor, era_metadata, vhr_sample_tensor, vhr_metadata

    


if __name__ == "__main__":
    era5_path = "datasets/regridded_era5.nc"
    era5_data = xr.open_dataset(era5_path, mask_and_scale=True)
    print("--- ERA5 Data Loaded ---")
    print(era5_data)
    
    vhr_path = "datasets/vhr-rea.nc"
    vhr_data = xr.open_dataset(vhr_path, mask_and_scale=True)
    print("--- VHR Data Loaded ---")
    print(vhr_data)

    italy_weather_dataset = ItalyWeatherDataset(era5_data, vhr_data)

    print(f"\nDataset length: {len(italy_weather_dataset)}")

    sample_idx = 0
    era5_sample_tensor, era5_metadata, vhr_sample_tensor, vhr_metadata = italy_weather_dataset[sample_idx]
    print(f"\n--- Sample {sample_idx} ---")
    print(f"--- ERA ---")
    print(f"Tensor shape: {era5_sample_tensor.shape}") # Expected: (2, num_rlat, num_rlon)
    print(f"Tensor dtype: {era5_sample_tensor.dtype}")
    print(f"Metadata: {era5_metadata}")
    print(f"--- VHR ---")
    print(f"Tensor shape: {vhr_sample_tensor.shape}") # Expected: (2, num_rlat, num_rlon)
    print(f"Tensor dtype: {vhr_sample_tensor.dtype}")
    print(f"Metadata: {vhr_metadata}")
    print(f"-----------------------------")

    plot_dataset_sample(era5_sample_tensor, vhr_sample_tensor)

    # # 5. Test with a DataLoader
    # print("\n--- Testing with DataLoader ---")
    # batch_size = 4
    # dataloader = DataLoader(italy_weather_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # # Iterate over a few batches
    # for i, (batch_tensors, batch_metadata) in enumerate(dataloader):
    #     print(f"\nBatch {i+1}:")
    #     print(f"  Batch Tensors shape: {batch_tensors.shape}") # Expected: (batch_size, 2, num_rlat, num_rlon)
    #     print(f"  Batch Tensors dtype: {batch_tensors.dtype}")
    #     print(f"  Metadata for first item in batch:")
    #     print(f"    valid_time: {batch_metadata['valid_time'][0]}") # Note: metadata fields are batched
    #     print(f"    expver: {batch_metadata['expver'][0]}")
    #     print(f"    number: {batch_metadata['number'][0]}") # This will be a tensor if it's numeric
    #     if i >= 1: # Print 2 batches
    #         break

    # # Example with a simple transform (e.g., normalization placeholder)
    # def example_transform(tensor):
    #     # Example: (tensor - mean) / std (calculate mean/std over the dataset or use fixed values)
    #     return (tensor * 2) - 1 # A simple scaling and shifting

    # transformed_dataset = ERA5XarrayDataset(era5_data, variables=['u10', 'v10'], transform=example_transform)
    # transformed_sample, _ = transformed_dataset[0]
    # print("\n--- Transformed Sample ---")
    # print(f"Transformed tensor (first few values of first channel):\n{transformed_sample[0, :2, :2]}")
    # original_sample, _ = era5_pytorch_dataset[0]
    # print(f"Original tensor (first few values of first channel):\n{original_sample[0, :2, :2]}")