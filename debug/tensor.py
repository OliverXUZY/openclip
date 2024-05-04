import torch

masks = torch.tensor([[ True, False,  True]])
masks = masks.squeeze()

print(masks.shape)

macs = torch.tensor([0.1,0.2,0.3])

res = masks*macs
print(res.shape, res)


