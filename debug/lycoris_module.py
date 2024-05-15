import sys
sys.path.insert(0, "/home/user/Documents/projects/openclip/")
sys.path.insert(0, "/home/zhuoyan/vision/openclip")
import numpy as np
import random

from tqdm import tqdm

import torch
import torch.nn as nn


from training.data import get_data
from training.params import parse_args

from lycoris import create_lycoris, LycorisNetwork


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

from open_clip.ada_vision_transformer import PlainMultiHeadAttention

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from open_clip.model import _build_vision_tower
from open_clip.ada_scheduler import ada_Scheduler, ada_SchedulerCfg


from training.distributed import is_master, init_distributed_device, broadcast_object
from training.precision import get_autocast

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

    ### replace new nn.multiheadattention with new module
    for module in model.visual.transformer.resblocks:
        # print(module.attn)
        new_module = PlainMultiHeadAttention(embed_dim=module.attn.embed_dim, num_heads=module.attn.num_heads)
        # print(new_module)
        # print("==============")
        new_module.set_parameters(module.attn)
        module.attn = new_module

    print(model.visual)

    
    print("args.precision", args.precision)
    autocast = get_autocast(args.precision)

    with autocast():
        # predict
        output = model(image=images)
    
    image_features = output['image_features'] if isinstance(output, dict) else output[0] # [bs, dim] [64, 512]
    
    print("in debug/model_forward: ", image_features.shape)

    for name, mod in model.visual.named_modules():
        # print(name)
        # print(mod)
        # print("=================")
        pass

    
    net = model.visual


    
    # LycorisNetwork.apply_preset({"target_name": [".*attn.*"]})
    LycorisNetwork.apply_preset({"target_module": ["ada_VisionTransformer"]})
    # LycorisNetwork.apply_preset({"target_module": ["MultiheadAttention"]})

    lycoris_net1 = create_lycoris(net, 1.0, linear_dim=64, linear_alpha=2.0, algo="lora")
    lycoris_net1.apply_to()
    lycoris_net1.cuda()

    print(f"#Modules of net1: {len(lycoris_net1.loras)}")

    num_total = sum(p.numel() for p in net.parameters())
    num_net1 = sum(p.numel() for p in lycoris_net1.parameters())
    print("Total params:", num_total)
    print("Net1 Params:", num_net1)
    print("net1/total: ", num_net1/num_total)


    for name, mod in lycoris_net1.named_modules():
        # print(name)
        # print(mod)
        # print("=================")
        pass

    



if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)

    main(args)
