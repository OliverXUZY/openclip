import sys
sys.path.insert(0, "/home/user/Documents/projects/openclip/")
import numpy as np
import torch




ckpt_path = "./logs/euler/ep100/epoch_last.pt"

preckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

print(preckpt.keys())
# print(preckpt['state_dict'].keys())
model_ckpt = preckpt['state_dict']


print(model_ckpt['visual.conv1.weight'][0][0])
print(model_ckpt['visual.transformer.resblocks.2.mlp.c_proj.weight'][0])
print(model_ckpt['transformer.resblocks.6.mlp.c_proj.weight'][0])
