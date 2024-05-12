
import sys
sys.path.insert(0, "/home/zhuoyan/vision/openclip")
import torch
import torch.nn as nn
from torchprofile import profile_macs

from ptflops import get_model_complexity_info



model = nn.Linear(in_features=10, out_features=20, bias=True)
inputs = torch.randn(1,10)


macs = profile_macs(model, inputs)

print("macs: ", macs)


print("==========================")

macs, params = get_model_complexity_info(model, (1, 10), as_strings=True, backend='pytorch',
                                           print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
