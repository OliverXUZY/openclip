
from typing import Callable, Optional, Sequence, Tuple
import torch
import torch.nn as nn
from .transformer import VisionTransformer, Transformer, ResidualAttentionBlock, QuickGELU, LayerNorm, _expand_token

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
        count_macs: Optional[bool] = False,
    ):
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
                all_false = (drop_block_mask == False).all()
                if all_false:
                    # if all false in count masks, skip all operations to avoid empty tensor into attn function
                    return q_x
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


class ada_Transformer(Transformer):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value,
            act_layer,
            norm_layer,
        )
        self.resblocks = nn.ModuleList([
            ada_ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, drop_block_masks = None, count_macs:Optional[bool] = False):
        """
        Args:
            x (torch.float16, [num_patchs (n_tokens), bs, dim]): input features.
            drop_block_masks (bool tensor, (bs, n)): masks for residual connections.
        """
        # print(x.shape)                  # [num_patchs (n_tokens), bs, dim] [50, 4(64), 768]
        for block_idx, r in enumerate(self.resblocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                # print("r: ", x.shape)
                # assert False
                assert isinstance(r, ada_ResidualAttentionBlock), "this vit should use customized Block"
                if drop_block_masks is None:
                    drop_block_mask = None
                else:
                    drop_block_mask = drop_block_masks[:, block_idx]
                    # print(drop_block_mask.shape, drop_block_mask)
                    # assert False
                x = r(x, attn_mask=attn_mask, drop_block_mask = drop_block_mask, count_macs = count_macs)
        # assert False
        return x



class ada_VisionTransformer(VisionTransformer):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'learnable',
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
    ):
        super().__init__(
            image_size,
            patch_size,
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value,
            attentional_pool,
            attn_pooler_queries,
            attn_pooler_heads,
            output_dim,
            patch_dropout,
            no_ln_pre,
            pos_embed_type,
            pool_type,
            final_ln_after_pool,
            act_layer,
            norm_layer,
            output_tokens,
        )
        self.transformer = ada_Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        ### save out macs
        self.macs = [
            0.02619428620948564,
            0.08115047614920953,
            0.08115047614920953,
            0.08115047614920953,
            0.0811504761492095,
            0.08115047614920956,
            0.08115047614920956,
            0.08115047614920945,
            0.08115047614920956,
            0.08115047614920956,
            0.08115047614920956,
            0.08115047614920945,
            0.08115047614920956
        ]
    
    def get_macs(self):
        return torch.tensor(self.macs).to(torch.float32)
    
    def forward(self, x: torch.Tensor, drop_block_masks=None, count_macs:Optional[bool] = False):
        """
        Args:
            x (torch.float16, [bs, c, h, w]): input features. [64, 3, 224, 224]
            drop_block_masks (bool tensor, (bs, n)): masks for residual connections.
        """
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # print(x.shape)                  # [num_patchs (n_tokens), bs, dim] [50, 4(64), 768]
        x = self.transformer(x, drop_block_masks = drop_block_masks, count_macs = count_macs)
        # print("after vit's tranasformer: ", x.shape)                  # [num_patchs (n_tokens), bs, dim] [50, 4(64), 768]
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens
        
        # print("after vit: ", pooled.shape)                  # [bs, fea_dim] [4, 512]
        # assert False
        return pooled
