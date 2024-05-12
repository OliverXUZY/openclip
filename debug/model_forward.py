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
    
    '''
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

    with autocast():
        # predict
        output = model(image=images)
    
    image_features = output['image_features'] if isinstance(output, dict) else output[0] # [bs, dim] [64, 512]
    
    print("in debug/model_forward: ", image_features.shape)
    '''

    ##### vision
    print("============================  vision =================================================================================")
    print('Loading model...')
    model_cfg = {
        'embed_dim': 512, 
        'quick_gelu': True, 
        'vision_cfg': {'image_size': 224, 'layers': 12, 'width': 768, 'patch_size': 32}, 
        # 'text_cfg': {'context_length': 77, 'vocab_size': 49408, 'width': 512, 'heads': 8, 'layers': 12}
    }

    visual = _build_vision_tower(**model_cfg, cast_dtype = None)
    visual.to(device="cuda")

    rng = random_seed(42)

    schdeuler_cfg = ada_SchedulerCfg()

    scheduler = ada_Scheduler(schdeuler_cfg)
    scheduler.to(device="cuda")
    
    testing_sche = True
    if testing_sche == False:
        drop_block_masks = torch.tensor(
            [True, False, True, True, True, False, True, True, True, False, True]
        ).unsqueeze(0).to("cuda")
        drop_block_masks = drop_block_masks.expand(4, -1)
        print("drop_block_masks: ", drop_block_masks.shape, drop_block_masks)

        output = visual(images, drop_block_masks = drop_block_masks, count_macs = True)
        # output = visual(images)
        print("in debug/visual_forward: ", output.shape, output[0, :20])
    else:
        dummy_latency = torch.tensor([0.5, 0.3, 0.8, 0.6]).to("cuda")
        output = visual(images, latency = dummy_latency, ada_scheduler_forward = scheduler.forward)







if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)

    main(args)