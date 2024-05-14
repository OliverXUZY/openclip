import torch
import torch.nn as nn
from peft import LoraConfig, LoraModel

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(32 * 8 * 8, 128)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.linear2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.linear1(x))
        x = x.unsqueeze(0)  # Adding sequence dimension for multihead attention
        x, _ = self.attention(x, x, x)
        x = x.squeeze(0)  # Removing sequence dimension after attention
        x = self.linear2(x)
        return x

def main():
    model = SimpleModel()
    # Define the LoRA configuration
    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["conv1", "conv2", "linear1", "attention", "linear2"],
        lora_dropout=0.1,
    )

    # Apply LoRA to the model
    lora_model = LoraModel(model, config, "default")

    # Check the modified model
    print(lora_model)

if __name__ == "__main__":
    main()
