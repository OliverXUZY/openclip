
import sys
sys.path.insert(0, "/home/zhuoyan/vision/openclip")
import numpy as np
import random
import torch
import torch.nn as nn
from open_clip.ada_scheduler import ada_Scheduler, ada_SchedulerCfg

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def main():
    rng = random_seed(42)

    schdeuler_cfg = ada_SchedulerCfg()

    scheduler = ada_Scheduler(schdeuler_cfg)
    print(scheduler)
    # assert False

    # Create a dummy input image of shape [1, 3, 32, 32]
    # 1 in the batch size, 3 for the number of channels (RGB), and 32x32 for the image dimensions
    dummy_img = torch.randn(2, 768)
    
    # Create a dummy latency scalar
    # Here we just use a single value, but it should be a float representing the latency
    dummy_latency = torch.tensor([0.5, 0.3])
    # print("dummy_latency: ", dummy_latency.shape)
    # print("dummy_latency: ", dummy_latency.view(-1,1).shape)
    
    # Pass the dummy image and latency through the model
    output = scheduler(dummy_img, dummy_latency)

    print(output, output.shape)



if __name__ == '__main__':
    main()