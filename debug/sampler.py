import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader

# Mock dataset with 10 classes, 100 samples each
num_classes = 3
samples_per_class = 10
total_samples = num_classes * samples_per_class

# Simulate dataset.targets with repeated class labels
dataset_targets = np.repeat(np.arange(num_classes), samples_per_class)

print("dataset_targets: ", dataset_targets)


# Number of samples per class to include in the sampler
k = 3

# Initialize the idxs array
idxs = np.zeros(total_samples)
print("idxs: ", idxs, len(idxs))

# Randomly select k indices from each class to include
for c in range(num_classes):
    class_mask = (dataset_targets == c)
    n = np.sum(class_mask)  # Number of samples in the class
    selection = np.zeros(n)
    selection[:k] = 1
    np.random.shuffle(selection)
    idxs[class_mask] = selection
    # print(f"class {c}")
    # print("idxs: ", idxs, len(idxs))

# Convert idxs to int and create the sampler
idxs = idxs.astype(int)

print("idxs: ", idxs, len(idxs))

print(" np.where(idxs == 1): ",  np.where(idxs == 1))

assert False
selected_indices = np.where(idxs == 1)[0]
sampler = SubsetRandomSampler(selected_indices)


# Mock DataLoader to illustrate the usage
# Normally, you'd use a dataset object compatible with PyTorch (like from torchvision.datasets)
batch_size = 10
data_loader = DataLoader(dataset_targets, batch_size=batch_size, sampler=sampler)

# Example loop to illustrate how data would be loaded during training
for data in data_loader:
    print("Batch of class labels:", data)
    # Normally here you would have images and labels as tensors
