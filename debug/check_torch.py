import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

print(torch.cuda.current_device())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

import torch.nn as nn


import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
model = nn.Linear(10, 2)  # Assuming you want to map from 10 features to 2 classes

# Generate random inputs and corresponding labels
inputs = torch.randn([4, 10])
labels = torch.tensor([0, 1, 0, 1])

# Send model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Also send inputs and labels to the same device
inputs = inputs.to(device)
labels = labels.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)  # lr is the learning rate

loss_fn = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()  # Initialize GradScaler
optimizer.zero_grad()  # Clear existing gradients
with torch.cuda.amp.autocast():  # Enable AMP
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
scaler.scale(loss).backward()  # Scale the loss and call backward
scaler.step(optimizer)  # Update the weights
scaler.update()  # Prepare the scaler for the next iteration



import torch
import torch.nn as nn

# Assuming conv1 is configured properly
conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1).cuda()  # Example parameters
x = torch.randn(64, 3, 224, 224).to(torch.float32).cuda()  # Example input tensor

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Forward pass
out = conv1(x)
print("Output shape from conv1:", out.shape)

# Backward pass
out.mean().backward()
print("Backward pass completed successfully.")
