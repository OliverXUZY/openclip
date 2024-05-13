import sys
sys.path.insert(0, "/home/user/Documents/projects/openclip/")
sys.path.insert(0, "/home/zhuoyan/vision/openclip")
import numpy as np
import random

from tqdm import tqdm

import torch


from training.data import get_data
from training.params import parse_args


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from open_clip.model import _build_vision_tower
from open_clip.ada_scheduler import ada_Scheduler, ada_SchedulerCfg


from training.distributed import is_master, init_distributed_device, broadcast_object
from training.precision import get_autocast
from src.utils import set_gpu

# Set the seed
torch.manual_seed(0)

def main(args):

    # fully initialize distributed device environment
    device = init_distributed_device(args)
    print("device: ", device)
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10
    
    images = torch.randn([4, 3, 224, 224])
    print("args.device", args.device)
    input_dtype = torch.float32
    images = images.to(device=args.device, dtype=input_dtype) # torch.float32 [bs, 3, 224, 224]
    
    
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        **model_kwargs,
    )

    
    print("args.precision", args.precision)
    autocast = get_autocast(args.precision)

    # with autocast():
    #     # predict
    #     output = model(image=images)
    
    # image_features = output['image_features'] if isinstance(output, dict) else output[0] # [bs, dim] [64, 512]
    
    # print("in debug/model_forward: ", image_features.shape)

    print("=================================== models ===================================")

    # print(model)

    check_gradient = False
    if check_gradient:
        print("=================================== parameters gradient ===================================")
        names = []
        for name, par in model.named_parameters():
            # if "logit_scale" in name:
            #     print(name)
            #     print(par.parameters())
            # print(name)
            # print(par.requires_grad)
            # print(par.grad)
            # names.append(name)
            pass

        # assert False
        # Freeze all the parameters in the model
        for param in model.parameters():
            param.requires_grad_(False)
        
        # Set requires_grad to True for all 'para' parameters and initialize them if necessary
        for name, param in model.named_parameters():
            if 'visual' in name or "logit_scale" in name:
                param.requires_grad_(True)
        
        
        for name, par in model.named_parameters():
            if par.requires_grad:
                print(name)
                print(par.requires_grad)
    
    print("=================================== optimizers ===================================")
    # create optimizer and scaler
    optimizer = None
    scaler = None



    
if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)

    main(args)