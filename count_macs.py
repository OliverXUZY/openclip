import os
import argparse

import torch
import numpy as np
from ptflops import get_model_complexity_info

from open_clip.model import _build_vision_tower

def input_constructor_factory(mask):
    def input_constructor(size):
        x = torch.empty(size).cuda()
        # print('x', x[None],x[None].shape)
        # print('mask', mask[None], mask[None].shape)
        # assert False
        return {'x': x[None], 'drop_block_masks': mask[None], 'count_macs': True}
    return input_constructor

def main(args):
    # load model
    print('Loading model...')
    model_cfg = {
        'embed_dim': 512, 
        'quick_gelu': True, 
        'vision_cfg': {'image_size': 224, 'layers': 12, 'width': 768, 'patch_size': 32}, 
        # 'text_cfg': {'context_length': 77, 'vocab_size': 49408, 'width': 512, 'heads': 8, 'layers': 12}
    }

    model = _build_vision_tower(**model_cfg, cast_dtype = None)
    model.to(device="cuda")


    # define branch masks
    n_knobs = model_cfg['vision_cfg']['layers']
    print('Number of knobs: {:d}'.format(n_knobs))
    print('Building masks...')
    masks = np.concatenate(
        [np.zeros((1, n_knobs)), np.tril(np.ones((n_knobs, n_knobs)))]
    )
    masks = torch.from_numpy(masks).bool()
    masks = masks.cuda()    
    
    # input size
    size = (3, 224, 224)

    # input_constructor = input_constructor_factory(masks)
    # input_constructor(size)
    # assert False

    print('Calculating per-block MACs breakdown...')
    all_macs = []
    for mask in masks:
        macs, _ = get_model_complexity_info(
            model, size,
            input_constructor=input_constructor_factory(mask),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        # print(f"mask: {mask}")
        # print(macs)
        # assert False

        all_macs.append(macs)
    
    print(f"actual macs: {all_macs}")
    # print(f"normed actual macs: {[i/557828196 for i in all_macs]}")

    # normalize relative to full model
    # all_macs = [macs / all_macs[-1] for macs in all_macs]
    assert False

    macs_breakdown = [all_macs[0]]
    for i in range(1, len(all_macs)):
        macs_breakdown.append(all_macs[i] - all_macs[i - 1])
    macs_breakdown = np.array(macs_breakdown, dtype=np.float32)

    print(list(macs_breakdown))
    print(f"sum {(macs_breakdown).sum()}")
    assert False
    
    os.makedirs(args.path, exist_ok=True)
    out_path = os.path.join(
        args.path, '{:s}_{:s}.npy'.format(args.arch, args.dataset)
    )
    np.save(out_path, macs_breakdown)
    print('Done!')

###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--arch', type=str, help='model architecture', default = 'resnet18',
        choices=('resnet18', 'resnet34', 'resnet50', 'resnet101', 'para_resnet18'),
    )
    parser.add_argument(
        '-d', '--dataset', type=str, help='dataset name', default = 'cifar100',
        choices=('cifar10', 'cifar100', 'imagenet'),
    )
    parser.add_argument('-p', '--path', type=str, help='output path')
    args = parser.parse_args()

    main(args)