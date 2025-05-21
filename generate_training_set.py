import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
 
import torch
from torch.utils.data import Dataset, DataLoader, random_split
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
    plt.title('ERA5 Vector Field Intensity Map')
    plt.xlabel('Pixel X-coordinate (width)')
    plt.ylabel('Pixel Y-coordinate (height)')
    plt.colorbar(label='Wind Magnitude')
 
    plt.subplot(1, 2, 2)
    plt.imshow(vhr_magnitude_numpy, cmap='viridis', origin='lower')
    plt.title('VHR Vector Field Intensity Map')
    plt.xlabel('Pixel X-coordinate (width)')
    plt.ylabel('Pixel Y-coordinate (height)')
    plt.colorbar(label='Wind Magnitude')
 
    plt.tight_layout()
    plt.show()
 
 
class ItalyWeatherDataset(Dataset):
    
    era5_variables = ["u10", "v10"]
    vhr_variables = ["U_10M", "V_10M"]
    
    def __init__(self, era5_dataset, vhr_dataset, transform=None, target_transform=None, 
                 era5_normalizer=None, vhr_normalizer=None):
        super().__init__()
        self.era5_dataset = era5_dataset
        self.vhr_dataset = vhr_dataset
        self.transform = transform
        self.target_transform = target_transform
        self.era5_normalizer = era5_normalizer
        self.vhr_normalizer = vhr_normalizer
 
        self.num_samples = len(self.era5_dataset.valid_time)
 
    def __len__(self):
        return self.num_samples
 
    def extract_region(self, data_tensor):
        patch_h, patch_w = (224, 224)  # TODO: FARE 224
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
    
        # Apply normalizers if provided
        if self.era5_normalizer is not None:
            era_sample_tensor = self.era5_normalizer.normalize(era_sample_tensor)
            
        if self.vhr_normalizer is not None:
            vhr_sample_tensor = self.vhr_normalizer.normalize(vhr_sample_tensor)
            
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
 
 
class Normalizer:
    """
    A class to handle normalization of tensors based on statistics computed from training data
    """
    def __init__(self, mean=None, std=None, dim=(0, 2, 3)):
        """
        Initialize normalizer with precomputed mean and std (or None to compute them later)
        
        Args:
            mean: Precomputed mean tensor
            std: Precomputed standard deviation tensor
            dim: Dimensions to compute statistics over (default: channels, height, width dimensions)
        """
        self.mean = mean
        self.std = std
        self.dim = dim  # Dimensions to compute stats over
        
    def compute_stats(self, data_loader):
        """
        Compute mean and standard deviation from a data loader
        
        Args:
            data_loader: DataLoader providing data to compute statistics from
        """
        print("Computing normalization statistics...")
        n_samples = 0
        mean_sum = 0
        var_sum = 0
        
        # First pass: compute mean
        for era_batch, _ in tqdm(data_loader, desc="Computing mean"):
            batch_size = era_batch.size(0)
            n_samples += batch_size
            mean_sum += era_batch.sum(dim=self.dim, keepdim=True)
        
        self.mean = mean_sum / n_samples
        
        # Second pass: compute std
        for era_batch, _ in tqdm(data_loader, desc="Computing std"):
            batch_size = era_batch.size(0)
            var_sum += ((era_batch - self.mean) ** 2).sum(dim=self.dim, keepdim=True)
            
        self.std = torch.sqrt(var_sum / n_samples)
        
        # Prevent division by zero
        self.std[self.std < 1e-8] = 1.0
        
        return self.mean, self.std
        
    def normalize(self, x):
        """
        Normalize input tensor using computed statistics
        
        Args:
            x: Input tensor to normalize
            
        Returns:
            Normalized tensor
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer statistics not computed or provided!")
        
        # Make sure dimensions match for broadcasting
        if x.dim() == 3:  # If single sample with [C, H, W]
            return (x - self.mean.squeeze(0)) / self.std.squeeze(0)
        else:  # If batch with [B, C, H, W]
            return (x - self.mean) / self.std
            
    def save(self, path):
        """Save normalizer statistics to file"""
        torch.save({
            'mean': self.mean,
            'std': self.std
        }, path)
        
    @classmethod
    def load(cls, path):
        """Load normalizer statistics from file"""
        stats = torch.load(path)
        return cls(mean=stats['mean'], std=stats['std'])
 
 
 
if __name__ == "__main__":
    OUTPUT_DIR = "datasets/normalized"

    train_ratio=0.7
    val_ratio=0.15
    test_ratio=0.15
    
    era5_path = "datasets/regridded_era5.nc"

    vhr_q1_path = "datasets/vhr_q1.nc"
    vhr_q2_path = "datasets/vhr_q2.nc"

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    era5_data = xr.open_dataset(era5_path, mask_and_scale=True)

    print(era5_data)

    vhr_data1 = xr.open_dataset(vhr_q1_path, mask_and_scale=True)
    vhr_data2 = xr.open_dataset(vhr_q2_path, mask_and_scale=True)

    vhr_data = xr.concat([vhr_data1, vhr_data2], dim="time")

    # Create dataset without normalization
    full_dataset = ItalyWeatherDataset(era5_data, vhr_data)

    # Split dataset indices
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    
    print(f"Dataset split: Train={train_size}, Validation={val_size}, Test={test_size}")
    
    # Create data loaders for computing normalization stats (only on training data)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        shuffle=False,  # No need to shuffle for computing stats
        num_workers=4
    )

    # Compute normalizers using only training data
    era5_normalizer = Normalizer()
    vhr_normalizer = Normalizer()
    
    # Extract ERA5 samples for computing stats
    era5_train_samples = []
    vhr_train_samples = []
    
    print("Extracting training samples for normalization...")
    for idx in tqdm(train_dataset.indices, desc="Collecting training samples"):
        era_sample, vhr_sample = full_dataset[idx]
        era5_train_samples.append(era_sample.unsqueeze(0))  # Add batch dimension
        vhr_train_samples.append(vhr_sample.unsqueeze(0))
    
    # Stack all samples
    era5_train_tensor = torch.cat(era5_train_samples, dim=0)
    vhr_train_tensor = torch.cat(vhr_train_samples, dim=0)
    
    # Compute normalization statistics
    era5_mean = era5_train_tensor.mean(dim=(0, 2, 3), keepdim=True)
    era5_std = era5_train_tensor.std(dim=(0, 2, 3), keepdim=True)
    
    vhr_mean = vhr_train_tensor.mean(dim=(0, 2, 3), keepdim=True)
    vhr_std = vhr_train_tensor.std(dim=(0, 2, 3), keepdim=True)
    
    # Set computed statistics in normalizers
    era5_normalizer.mean = era5_mean
    era5_normalizer.std = era5_std
    
    vhr_normalizer.mean = vhr_mean
    vhr_normalizer.std = vhr_std
    
    print("ERA5 Stats - Mean:", era5_mean.squeeze(), "Std:", era5_std.squeeze())
    print("VHR Stats - Mean:", vhr_mean.squeeze(), "Std:", vhr_std.squeeze())
    
    # Save normalizers
    era5_normalizer.save(os.path.join(OUTPUT_DIR, 'era5_normalizer.pt'))
    vhr_normalizer.save(os.path.join(OUTPUT_DIR, 'vhr_normalizer.pt'))
    
    # Create normalized dataset splits with appropriate normalizers
    normalized_train_dataset = ItalyWeatherDataset(
        era5_data, vhr_data, 
        era5_normalizer=era5_normalizer, 
        vhr_normalizer=vhr_normalizer
    )
    
    # Create dictionary with dataset splits
    dataset_splits = {
        'train': torch.utils.data.Subset(normalized_train_dataset, train_dataset.indices),
        'val': torch.utils.data.Subset(normalized_train_dataset, val_dataset.indices),
        'test': torch.utils.data.Subset(normalized_train_dataset, test_dataset.indices)
    }
    
    # Save split indices for future reference
    torch.save({
        'train_indices': train_dataset.indices,
        'val_indices': val_dataset.indices,
        'test_indices': test_dataset.indices
    }, os.path.join(OUTPUT_DIR, 'dataset_splits.pt'))
    
    # Save normalized training data
    era5_train_normalized = []
    vhr_train_normalized = []
    
    print("Normalizing and collecting training samples...")
    for idx in tqdm(train_dataset.indices, desc="Processing training samples"):
        era_sample, vhr_sample = normalized_train_dataset[idx]
        era5_train_normalized.append(era_sample.unsqueeze(0))
        vhr_train_normalized.append(vhr_sample.unsqueeze(0))
    
    era5_train_norm_tensor = torch.cat(era5_train_normalized, dim=0)
    vhr_train_norm_tensor = torch.cat(vhr_train_normalized, dim=0)
    
    # Save preprocessed tensors
    torch.save({
        'era5': era5_train_norm_tensor,
        'vhr': vhr_train_norm_tensor
    }, os.path.join(OUTPUT_DIR, 'normalized_train_data.pt'))
    
    # Process validation data (without normalization stats computation)
    print("Processing validation data...")
    era5_val_normalized = []
    vhr_val_normalized = []
    
    for idx in tqdm(val_dataset.indices, desc="Processing validation samples"):
        era_sample, vhr_sample = normalized_train_dataset[idx]
        era5_val_normalized.append(era_sample.unsqueeze(0))
        vhr_val_normalized.append(vhr_sample.unsqueeze(0))
    
    era5_val_norm_tensor = torch.cat(era5_val_normalized, dim=0)
    vhr_val_norm_tensor = torch.cat(vhr_val_normalized, dim=0)
    
    torch.save({
        'era5': era5_val_norm_tensor,
        'vhr': vhr_val_norm_tensor
    }, os.path.join(OUTPUT_DIR, 'normalized_val_data.pt'))
    
    # Process test data (without normalization stats computation)
    print("Processing test data...")
    era5_test_normalized = []
    vhr_test_normalized = []
    
    for idx in tqdm(test_dataset.indices, desc="Processing test samples"):
        era_sample, vhr_sample = normalized_train_dataset[idx]
        era5_test_normalized.append(era_sample.unsqueeze(0))
        vhr_test_normalized.append(vhr_sample.unsqueeze(0))
    
    era5_test_norm_tensor = torch.cat(era5_test_normalized, dim=0)
    vhr_test_norm_tensor = torch.cat(vhr_test_normalized, dim=0)
    
    torch.save({
        'era5': era5_test_norm_tensor,
        'vhr': vhr_test_norm_tensor
    }, os.path.join(OUTPUT_DIR, 'normalized_test_data.pt'))

    print("\nNormalized datasets prepared successfully!")
    
    # Load the normalizers to verify
    era5_normalizer = Normalizer.load(os.path.join(OUTPUT_DIR, 'era5_normalizer.pt'))
    vhr_normalizer = Normalizer.load(os.path.join(OUTPUT_DIR, 'vhr_normalizer.pt'))
    
    print("\nLoaded ERA5 normalizer - Mean:", era5_normalizer.mean.squeeze(), "Std:", era5_normalizer.std.squeeze())
    print("Loaded VHR normalizer - Mean:", vhr_normalizer.mean.squeeze(), "Std:", vhr_normalizer.std.squeeze())