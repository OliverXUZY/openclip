import sys
sys.path.insert(0, "/home/user/Documents/projects/openclip/")
import torch
import torch.nn as nn
from peft import LoraConfig, LoraModel, get_peft_model
from open_clip.ada_vision_transformer import PlainMultiHeadAttention

from lycoris import create_lycoris, LycorisNetwork


# Set the seed
torch.manual_seed(0)
# Create a generator with a specific seed
g = torch.Generator()
g.manual_seed(0)
replace = True

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(32 * 8 * 8, 128)
        self.linear2 = nn.Linear(128, 10)
        
        nnAttn = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        if replace:
            self.attention = PlainMultiHeadAttention(embed_dim=128, num_heads=4)
            self.attention.set_parameters(nnAttn)
        else:
            print("original")
            self.attention = nnAttn
        
        # for name, par in self.attention.named_parameters():
        #     print(name)
        #     print(par.shape, type(par), par[0])

        
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

    print(model)


    net = model
    # LycorisNetwork.apply_preset({"target_module": ["PlainMultiHeadAttention"]})
    LycorisNetwork.apply_preset({"target_module": ["SimpleModel"]})

    lycoris_net1 = create_lycoris(net, 1.0, linear_dim=64, linear_alpha=2.0, algo="lora")
    lycoris_net1.apply_to()
    lycoris_net1.cuda()

    print(f"#Modules of net1: {len(lycoris_net1.loras)}")

    num_total = sum(p.numel() for p in net.parameters())
    num_net1 = sum(p.numel() for p in lycoris_net1.parameters())
    print("Total params:", num_total)
    print("Net1 Params:", num_net1)
    print("net1/total: ", num_net1/num_total)

    print(lycoris_net1.loras)

    for name, mod in lycoris_net1.named_modules():
        print(name)
        print(mod)
        print("=================")
        pass
    

    return

    # Define the LoRA configuration
    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["conv1", "conv2", "linear1", "linear2", "qkv", "proj"],
        lora_dropout=0.1,
    )

    # images = torch.randn((1, 1, 8, 8), generator = g)  # Batch size of 1, 1 channel, 8x8 image
    # print(images[0,0,:4,:4])
    # output = model(images)
    # print(output[0])

    # Apply LoRA to the model
    lora_model = LoraModel(model, config, "default")

    # for name, param in lora_model.named_parameters():
    #     if param.requires_grad:
    #         name += " === grad"
    #     print(name)
    # print("___________________________________________========================")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         name += " === grad"
    #     print(name)
    # # return

    lora_model.merge_adapter()

    model_st = model.state_dict()
    # print(model)

    lora_st = lora_model.state_dict()
    print(model_st['conv1.base_layer.weight'][0])

    print("=================================================================")
    # print(lora_model)


    print(lora_st['model.conv1.base_layer.weight'][0])


    # Check the modified model
    
    

    # print(model)



if __name__ == "__main__":
    main()
