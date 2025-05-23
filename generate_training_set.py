import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import xarray as xr
from tqdm import tqdm

SAVE = os.getenv(key="SAVE", default=0)
 
class ItalyWeatherDataset(Dataset):
    ERA5_VARIABLES = ["u10", "v10"]
    VHR_VARIABLES = ["U_10M", "V_10M"]
    
    def __init__(self, era5_dataset, vhr_dataset, era5_normalizer=None, vhr_normalizer=None):
        super().__init__()
        self.era5_dataset = era5_dataset
        self.vhr_dataset = vhr_dataset
        self.era5_normalizer = era5_normalizer
        self.vhr_normalizer = vhr_normalizer

        # ensure time coordinates are aligned
        if len(self.era5_dataset.valid_time) != len(self.vhr_dataset.time):
            raise ValueError("ERA5 and VHR datasets have different number of time steps.")
        self.num_samples = len(self.era5_dataset.valid_time)
 
    def __len__(self):
        return self.num_samples
 
    def extract_region(self, data_tensor):
        patch_h, patch_w = (224, 224)
        _, h, w = data_tensor.shape
        x = (w - patch_w) // 2
        y = (h - patch_h) // 2
        return data_tensor[:, y:y+patch_h, x:x+patch_w]
 
    def __getitem__(self, idx):
        if not 0 <= idx < self.num_samples:
            raise IndexError(f"Index {idx} is out of bounds for dataset with size {self.num_samples}")

        # processing ERA5 data
        era_sample_slice = self.era5_dataset.isel(valid_time=idx)
        era_data_arrays = [era_sample_slice[var_name].values for var_name in self.ERA5_VARIABLES]
        era_stacked_data_np = np.stack(era_data_arrays, axis=0)
        era_sample_tensor = torch.from_numpy(era_stacked_data_np).float()
        era_sample_tensor = self.extract_region(era_sample_tensor)
        if self.era5_normalizer is not None: era_sample_tensor = self.era5_normalizer.normalize(era_sample_tensor)
        
        # processing VHR data
        vhr_sample_slice = self.vhr_dataset.isel(time=idx)
        vhr_data_arrays = [vhr_sample_slice[var_name].values for var_name in self.VHR_VARIABLES]
        vhr_stacked_data_np = np.stack(vhr_data_arrays, axis=0)
        vhr_sample_tensor = torch.from_numpy(vhr_stacked_data_np).float()
        vhr_sample_tensor = self.extract_region(vhr_sample_tensor)
        if self.vhr_normalizer is not None: vhr_sample_tensor = self.vhr_normalizer.normalize(vhr_sample_tensor)

        return era_sample_tensor, vhr_sample_tensor
 
 
class MinMaxNormalizer:
    def __init__(self, min_val=None, max_val=None, feature_range=(0, 1)):
        self.min_val = min_val
        self.max_val = max_val
        self.feature_range = feature_range

    def compute_stats(self, data_tensor):
        print("Computing min-max normalization statistics...")
        if data_tensor.dim() != 4: raise ValueError("Input data_tensor must be 4D (N, C, H, W).")
        
        # calculate min/max per channel across (N, H, W) dimensions
        self.min_val = data_tensor.amin(dim=(0, 2, 3), keepdim=True)
        self.max_val = data_tensor.amax(dim=(0, 2, 3), keepdim=True)
        # adjust max_val where min_val == max_val to prevent division by zero
        diff = self.max_val - self.min_val
        epsilon = 1e-7
        self.max_val[diff < epsilon] = self.min_val[diff < epsilon] + epsilon
        
        print(f"Computed min: {self.min_val.squeeze()}")
        print(f"Computed max: {self.max_val.squeeze()}")
        return self.min_val, self.max_val
        
    def normalize(self, x):
        if self.min_val is None or self.max_val is None: raise ValueError("MinMaxNormalizer statistics not computed or provided!")
        
        min_v = self.min_val
        max_v = self.max_val

        if x.dim() == 3:  # if single sample with [C, H, W]
            min_v = self.min_val.squeeze(0) # Becomes [C, 1, 1]
            max_v = self.max_val.squeeze(0) # Becomes [C, 1, 1]

        # normalize to [0, 1] (epsilon case is already handled in compute_stats)
        x_normalized = (x - min_v) / (max_v - min_v)
        
        # scale to the feature range
        f_min, f_max = self.feature_range # in our case the feature range is [0, 1]
        return x_normalized if (f_min == 0 and f_max == 1) else x_normalized * (f_max - f_min) + f_min
            
    def save(self, path):
        torch.save({'min_val': self.min_val, 'max_val': self.max_val, 'feature_range': self.feature_range}, path)
        
    @classmethod
    def load(cls, path):
        stats = torch.load(path)
        return cls(min_val=stats['min_val'], max_val=stats['max_val'], feature_range=stats.get('feature_range', (0, 1)))
 

def save_split_data(dataset_subset, split_name, output_dir):
    print(f"Normalizing, collecting, and saving {split_name} samples...")
    era5_normalized_list = []
    vhr_normalized_list = []
    for idx in tqdm(dataset_subset.indices, desc=f"Processing {split_name} samples"):
        era_sample, vhr_sample = dataset_subset.dataset[idx]
        era5_normalized_list.append(era_sample.unsqueeze(0))
        vhr_normalized_list.append(vhr_sample.unsqueeze(0))
    
    era5_norm_tensor = torch.cat(era5_normalized_list, dim=0)
    vhr_norm_tensor = torch.cat(vhr_normalized_list, dim=0)
    
    torch.save({'era5': era5_norm_tensor, 'vhr': vhr_norm_tensor}, os.path.join(output_dir, f'normalized_{split_name}_data.pt'))
    print(f"Saved {split_name} data: ERA5 shape {era5_norm_tensor.shape}, VHR shape {vhr_norm_tensor.shape}")
    print(f"{split_name} ERA5 min: {era5_norm_tensor.min()}, max: {era5_norm_tensor.max()}")
    print(f"{split_name} VHR min: {vhr_norm_tensor.min()}, max: {vhr_norm_tensor.max()}")
 
 
if __name__ == "__main__":
    OUTPUT_DIR = "datasets/normalized_minmax"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_ratio=0.7
    val_ratio=0.15
    test_ratio=0.15
    
    era5_path = "datasets/regridded_era5.nc"
    vhr_path = "datasets/vhr-rea.nc"

    # vhr_q1_path = "datasets/vhr_q1.nc"
    # vhr_q2_path = "datasets/vhr_q2.nc"
    # vhr_data1 = xr.open_dataset(vhr_q1_path, mask_and_scale=True)
    # vhr_data2 = xr.open_dataset(vhr_q2_path, mask_and_scale=True)
    # vhr_data = xr.concat([vhr_data1, vhr_data2], dim="time")
    
    print("Loading datasets...")
    era5_data = xr.open_dataset(era5_path, mask_and_scale=True)
    vhr_data = xr.open_dataset(vhr_path, mask_and_scale=True)
    print(f"ERA5 data loaded: {era5_data.dims}")
    print(f"VHR data loaded: {vhr_data.dims}")

    full_dataset = ItalyWeatherDataset(era5_data, vhr_data)

    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # adjust test_size to account for rounding
    if train_size + val_size + test_size != dataset_size: test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    
    print(f"Dataset split: Train={len(train_dataset.indices)}, Validation={len(val_dataset.indices)}, Test={len(test_dataset.indices)}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print("Extracting training samples for normalization...")
    era5_train_samples = []
    vhr_train_samples = []
    for idx in tqdm(train_dataset.indices, desc="Collecting training samples"):
        era_sample, vhr_sample = full_dataset[idx] 
        era5_train_samples.append(era_sample.unsqueeze(0))
        vhr_train_samples.append(vhr_sample.unsqueeze(0))
    
    era5_train_tensor = torch.cat(era5_train_samples, dim=0)
    vhr_train_tensor = torch.cat(vhr_train_samples, dim=0)
    
    # compute normalization statistics
    era5_minmax_normalizer = MinMaxNormalizer(feature_range=(0, 1))
    era5_minmax_normalizer.compute_stats(era5_train_tensor)
    print("ERA5 Stats - Min:", era5_minmax_normalizer.min_val.squeeze(), "Max:", era5_minmax_normalizer.max_val.squeeze())
    if SAVE: era5_minmax_normalizer.save(os.path.join(OUTPUT_DIR, 'era5_minmax_normalizer.pt'))
    
    vhr_minmax_normalizer = MinMaxNormalizer(feature_range=(0, 1))
    vhr_minmax_normalizer.compute_stats(vhr_train_tensor)
    print("VHR Stats - Min:", vhr_minmax_normalizer.min_val.squeeze(), "Max:", vhr_minmax_normalizer.max_val.squeeze())
    if SAVE: vhr_minmax_normalizer.save(os.path.join(OUTPUT_DIR, 'vhr_minmax_normalizer.pt'))
    
    # dataset that applies the normalization based on stats from the training splilt
    normalized_full_dataset = ItalyWeatherDataset(era5_data, vhr_data, era5_normalizer=era5_minmax_normalizer, vhr_normalizer=vhr_minmax_normalizer)
    
    dataset_splits = {
        'train': torch.utils.data.Subset(normalized_full_dataset, train_dataset.indices),
        'val': torch.utils.data.Subset(normalized_full_dataset, val_dataset.indices),
        'test': torch.utils.data.Subset(normalized_full_dataset, test_dataset.indices)
    }
    
    if SAVE:
        torch.save({'train_indices': train_dataset.indices, 'val_indices': val_dataset.indices, 'test_indices': test_dataset.indices},
                   os.path.join(OUTPUT_DIR, 'dataset_splits.pt'))
        save_split_data(dataset_splits['train'], 'train', OUTPUT_DIR)
        save_split_data(dataset_splits['val'], 'val', OUTPUT_DIR)
        save_split_data(dataset_splits['test'], 'test', OUTPUT_DIR)
        print("\nNormalized datasets saved successfully!")

