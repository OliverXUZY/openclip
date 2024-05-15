
import json
import torch
from pprint import pprint
def save_json(data, file_path, indent = 4):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent = indent)

## num heads = 12, num_layers = 12

# all_macs = [115605504, 473753904, 831902304, 1190050704, 1548199104, 1906347504, 
#                 2264495904, 2622644304, 2980792704, 3338941104, 3697089504, 4055237904, 4413386304]

all_macs = [469845504, 824085504, 1178325504, 1532565504, 1886805504, 2241045504, 2595285504, 2949525504, 3303765504, 3658005504, 4012245504, 4366485504]
# all_macs = [macs / 10000000 for macs in all_macs]

all_macs = [macs / all_macs[-1] for macs in all_macs]
print(all_macs)
macs_breakdown = [all_macs[0]]
for i in range(1, len(all_macs)):
    macs_breakdown.append(all_macs[i] - all_macs[i - 1])

    
# macs_breakdown = torch.Tensor(macs_breakdown).to(torch.float32)
# print(macs_breakdown[-1] - macs_breakdown[-2])

print(macs_breakdown)
print(macs_breakdown.sum())

# save_json(macs_breakdown, "clip_ViT-B-32-quickgelu.json")


