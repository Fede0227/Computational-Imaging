import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch.utils.data import DataLoader, TensorDataset
from generate_training_set import plot_dataset_sample

dataset_path = "datasets/preprocessed_italy_weather.pt"

print(f"\n--- Loading preprocessed data from {dataset_path} ---")
preprocessed_data = torch.load(dataset_path)
era5_tensors = preprocessed_data['era5']
vhr_tensors = preprocessed_data['vhr']
print(f"Loaded ERA5 tensor shape: {era5_tensors.shape}")
print(f"Loaded VHR tensor shape: {vhr_tensors.shape}")

preprocessed_tensor_dataset = TensorDataset(era5_tensors, vhr_tensors)
print(f"Created TensorDataset with {len(preprocessed_tensor_dataset)} samples.")

# Example: Get a sample from the TensorDataset
sample_idx = 0
era_sample_from_tensor_ds, vhr_sample_from_tensor_ds = preprocessed_tensor_dataset[sample_idx]
print(f"\nSample {sample_idx} from TensorDataset:")
print(f"ERA5 sample shape: {era_sample_from_tensor_ds.shape}")
print(f"VHR sample shape: {vhr_sample_from_tensor_ds.shape}")

# Plot this sample
plot_dataset_sample(
  era_sample_from_tensor_ds,
  vhr_sample_from_tensor_ds,
)

data_loader = DataLoader(preprocessed_tensor_dataset, batch_size=4, shuffle=True)
era_batch, vhr_batch = next(iter(data_loader))
print(f"\nBatch from DataLoader (TensorDataset):")
print(f"ERA5 batch shape: {era_batch.shape}")
print(f"VHR batch shape: {vhr_batch.shape}")
