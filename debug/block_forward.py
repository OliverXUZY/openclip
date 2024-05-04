import sys
sys.path.insert(0, "/home/user/Documents/projects/openclip/")

from tqdm import tqdm
import numpy as np
import random

import torch
import torch.nn as nn

from training.data import get_data
from training.params import parse_args

from typing import Callable, Optional, Sequence, Tuple


from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from training.distributed import is_master, init_distributed_device, broadcast_object
from training.precision import get_autocast

# Set the seed
def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
random_seed()

from open_clip.transformer import ResidualAttentionBlock, QuickGELU, LayerNorm
from open_clip.ada_vision_transformer import ada_ResidualAttentionBlock
'''
class ada_ResidualAttentionBlock(ResidualAttentionBlock):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__(
            d_model,
            n_head,
            mlp_ratio,
            ls_init_value,
            act_layer,
            norm_layer,
            is_cross_attention,
        )
        self.attn = nn.MultiheadAttention(d_model, n_head)

    # TODO: not implement mask yet
    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            q_x (float tensor, (seq_len, bs, dim)): feature maps.  [50 (7*7+1), 4 (or 64), 768]
            drop_block_mask (bool tensor, (bs,)): mask for residual connection.
            drop_head_mask (bool tensor, (bs,)): mask for dropping attention head.
        """
        
        # print("q_x: ", q_x.shape)          # [num_patchs (n_tokens), bs, dim] [50, 4(64), 768]
        # print("k_x: ", k_x)
        # print("v_x: ", v_x)
        # assert False

        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        


        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]


    def forward(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        drop_block_mask: Optional[torch.Tensor] = None,
        drop_head_mask: Optional[torch.Tensor] = None,
        count_macs = False,
    ):
        print("================ count_macs =====================: ", count_macs)
        """
        Args:
            q_x (float tensor, (seq_len, bs, dim)): feature maps.  [50 (7*7+1), 4 (or 64), 768]
            drop_block_mask (bool tensor, (bs,)): mask for residual connection.
            drop_head_mask (bool tensor, (bs,)): mask for dropping attention head.
        """
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        
        if drop_block_mask is None:
            x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
            x = x + self.ls_2(self.mlp(self.ln_2(x)))
        else:
            assert k_x is None and v_x is None, "Only implement for self attn."
            if count_macs:
                res1 = self.ls_1(self.attention(q_x=self.ln_1(q_x[:,drop_block_mask,:]), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
                q_x[:,drop_block_mask,:] = q_x[:,drop_block_mask,:] + res1
                x = q_x

                res2 = self.ls_2(self.mlp(self.ln_2(x[:,drop_block_mask,:])))
                x[:,drop_block_mask,:] = x[:,drop_block_mask,:] + res2
            else:
                # Check if drop_block_mask is of boolean type
                if drop_block_mask.dtype == torch.bool:
                    # Convert to torch.float32
                    drop_block_mask = drop_block_mask.to(torch.float32)

                mask_reshaped = drop_block_mask.view(1, q_x.shape[1], 1)  # Reshape to (1, bs, 1) for broadcasting
                mask_tensor = mask_reshaped.float().expand_as(q_x) # broadcast mask to the shape of x and convert to float tensor

                res1 = self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
                q_x = q_x + res1 * mask_tensor
                x = q_x

                res2 = self.ls_2(self.mlp(self.ln_2(x)))
                x = x + res2 * mask_tensor
        # print("x: ", x.shape)              # [num_patchs (n_tokens), bs, dim] [50, 4(64), 768]
        # assert False
        return x

'''


def main():
    block  = ada_ResidualAttentionBlock(
        d_model = 6, # 768,
        n_head = 2, # 12,
        mlp_ratio = 4.0,
        ls_init_value = None,
        act_layer = QuickGELU,
        norm_layer = LayerNorm,
        is_cross_attention  = False,
    )
    input_dtype = torch.float32
    block = block.to("cuda")

    # features = torch.randn([50, 4, 768])
    features = torch.randn([2, 4, 6])
    # print(features)
    feature_ori = features.clone().to("cuda")  # This creates a new tensor that is a copy of q_x
    
    features = features.to(device="cuda", dtype=input_dtype) # torch.float32 

    # mask = torch.tensor([True, False, True, True]).to("cuda")
    mask = torch.tensor([False, False, False, False]).to("cuda")
    # Check if all elements are False
    # all_false = (mask == False).all()
    # if all_false:
    #     print("ifif")

    # print("all_false: ", all_false)  # This will print 'True' if all elements are False
    print(mask.dtype, mask, mask.shape)


    # mask_x = features[:,mask,:]
    # print("1:==============", mask_x)
    # print(mask_x.shape)

    print("features.device: ", features.device)

    ### drop block masks
    output = block(features, drop_block_mask = mask, count_macs = True)
    # output = block(features, drop_block_mask = mask, count_macs = False)

    # print("in debug/block_forward: ", output.shape)

    print(feature_ori - output)
    

    ### drop attention masks
    # output = block(features)

    # print("in debug/block_forward: ", output.shape)

    # print(feature_ori - output)


if __name__ == "__main__":
    main()
