
from typing import Callable, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .transformer import VisionTransformer, Transformer, ResidualAttentionBlock, QuickGELU, LayerNorm, _expand_token

class PlainMultiHeadAttention(nn.Module):
    def __init__(
            self,
            embed_dim=1024,
            num_heads=16,
            dropout=0.,
            bias=True,
            kdim=None,
            vdim=None,
            batch_first=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            assert NotImplementedError
        else:
            self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.scaled_dot_product_attention = F.scaled_dot_product_attention

        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def init_weights(self):
        pass

    def forward(
            self,
            query,
            key,
            value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False):

        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")
        is_batched = query.dim() == 3
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        E = query.size(-1)
        qkv = self.qkv(query)
        qkv = qkv.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=F._none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=q.dtype,
            check_other=False,
        )

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

        dropout_p = self.dropout if self.training else 0.

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        attn_output = self.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        return attn_output, None

    def set_parameters(self, torch_tgt_module):
        assert isinstance(torch_tgt_module, nn.MultiheadAttention)
        assert self.embed_dim == torch_tgt_module.embed_dim
        assert self.batch_first == torch_tgt_module.batch_first
        assert self.dropout == torch_tgt_module.dropout
        assert self.head_dim == torch_tgt_module.head_dim
        assert self.num_heads == torch_tgt_module.num_heads
        assert self.kdim == torch_tgt_module.kdim
        assert self.vdim == torch_tgt_module.vdim
        self.qkv.weight.data = torch_tgt_module.in_proj_weight.data
        self.qkv.bias.data = torch_tgt_module.in_proj_bias.data
        self.proj.weight.data = torch_tgt_module.out_proj.weight.data
        self.proj.bias.data = torch_tgt_module.out_proj.bias.data


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

        # if drop_block_mask is not None:
        #     print("drop_block_mask.device: ", drop_block_mask.device)
        #     print("drop_block_mask: ", drop_block_mask)
        
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
                    # print("convert bool to float, might affect gradient flow in training")
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
    
    def _gumbel_sigmoid(self, logits, tau=1, hard=False, eps=1e-10):
        # Sample Gumbel noise
        gumbels1 = -torch.empty_like(logits).exponential_().log()  # Sample from Gumbel(0, 1)
        gumbels2 = -torch.empty_like(logits).exponential_().log()  # Sample from Gumbel(0, 1)
        # print(gumbels.shape)
        # assert False
        # Add Gumbel noise to logits
        noisy_logits = (logits + gumbels1 - gumbels2) / tau  # Apply temperature
        # Apply sigmoid to get probabilities in (0, 1)
        y_soft = torch.sigmoid(noisy_logits)
        
        if hard:
            # Hard thresholding to 0 or 1, but in a way that gradients can flow through y_soft
            y_hard = (y_soft > 0.5).float()
            # Use straight-through estimator to make the operation differentiable
            y = y_hard - y_soft.detach() + y_soft
        else:
            y = y_soft
        
        return y
    
    def forward_blockmask(
            self, x: torch.Tensor, 
            attn_mask: Optional[torch.Tensor] = None, 
            drop_block_masks = None, 
            count_macs:Optional[bool] = False,
        ):
        """
        Args:
            x (torch.float16, [num_patchs (n_tokens), bs, dim]): input features.
            drop_block_masks (bool tensor, (bs, n)): masks for residual connections.
        """
        # print(x.shape)                  # [num_patchs (n_tokens), bs, dim] [50, 4(64), 768]
        mask_idx = 0
        for block_idx, r in enumerate(self.resblocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                # print("r: ", x.shape)
                # assert False
                assert isinstance(r, ada_ResidualAttentionBlock), "this vit should use customized Block"
                if drop_block_masks is None or block_idx == 0:
                    ## always enable first block for now
                    drop_block_mask = None
                else:
                    drop_block_mask = drop_block_masks[:, mask_idx]
                    mask_idx += 1
                    # print(drop_block_mask.shape, drop_block_mask)
                    # assert False
                x = r(x, attn_mask=attn_mask, drop_block_mask = drop_block_mask, count_macs = count_macs)
        # assert False
        return x, drop_block_masks

    def forward_block_scheduler(
            self, x: torch.Tensor, 
            attn_mask: Optional[torch.Tensor] = None, 
            latency: Optional[torch.Tensor] = None,
            ada_scheduler_forward: Callable = None,
        ):
        """
        Args:
            x (torch.float16, [num_patchs (n_tokens), bs, dim]): input features.
            latency: (torch.float32, (bs, )).
            ada_scheduler_forward: Callable
        """
        # print(x.shape)                  # [num_patchs (n_tokens), bs, dim] [50, 4(64), 768]
        assert latency is not None, "must provide latency"
        mask_idx = 0
        for block_idx, r in enumerate(self.resblocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                # print("r: ", x.shape)
                # assert False
                assert isinstance(r, ada_ResidualAttentionBlock), "this vit should use customized Block"
                if block_idx == 0:
                    ## always enable first block for now
                    drop_block_mask = None
                    ada_sche_x = x.clone().permute(1, 0, 2)  # LND -> NLD
                    ada_pooled = ada_sche_x[:, 0]
                    # print(ada_sche_x.shape)
                    # print(ada_pooled.shape)
                    # assert False
                    logits = ada_scheduler_forward(ada_pooled, latency)
                    drop_block_masks = self._gumbel_sigmoid(logits, hard = True)
                    # print("generate drop_block_masks: ", drop_block_masks.shape)
                    # print(drop_block_masks)
                else:
                    drop_block_mask = drop_block_masks[:, mask_idx]
                    mask_idx += 1
                    
                x = r(x, attn_mask=attn_mask, drop_block_mask = drop_block_mask, count_macs = False)
        # assert False
        return x, drop_block_masks

    def forward(
            self, x: torch.Tensor, 
            attn_mask: Optional[torch.Tensor] = None, 
            drop_block_masks = None, 
            count_macs:Optional[bool] = False,
            latency: Optional[torch.Tensor] = None,
            ada_scheduler_forward: Callable = None,
        ):
        """
        Args:
            x (torch.float16, [num_patchs (n_tokens), bs, dim]): input features.
            drop_block_masks (bool tensor, (bs, n)): masks for residual connections.
            latency: (torch.float32, (bs, )).
            ada_scheduler_forward: Callable
        """
        if latency is not None:
            assert drop_block_masks is None, "provide block mask and latency both!"
            x, drop_block_masks = self.forward_block_scheduler(
                x=x, attn_mask=attn_mask, latency = latency, ada_scheduler_forward = ada_scheduler_forward
            )
        else:
            # assert drop_block_masks is not None, "must provide block mask if no latency requirement!"
            x, drop_block_masks = self.forward_blockmask(
                x=x, attn_mask=attn_mask, drop_block_masks = drop_block_masks, count_macs = count_macs
            )
        return x, drop_block_masks



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
    
    def forward(
            self, 
            x: torch.Tensor, 
            drop_block_masks=None, 
            count_macs:Optional[bool] = False,
            latency: Optional[torch.Tensor] = None,
            ada_scheduler_forward: Callable = None,
        ):
        """
        Args:
            x (torch.float16, [bs, c, h, w]): input features. [64, 3, 224, 224]
            drop_block_masks (bool tensor, (bs, n)): masks for residual connections.
            latency: (torch.float32, (bs, )).
            ada_scheduler_forward: Callable
        """
        # print("Input to conv:", x.shape, x.dtype)
        # print("Weights:", self.conv1.weight.shape, self.conv1.weight.dtype)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # print("Input to conv:", x.shape, x.dtype)
        # print("Weights:", self.conv1.weight.shape, self.conv1.weight.dtype)

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # print("x.shape: ", x.shape, x.dtype)              # [num_patchs (n_tokens), bs, dim] [50, 4(64), 768]
        x, drop_block_masks = self.transformer(
            x, 
            drop_block_masks = drop_block_masks, 
            count_macs = count_macs,
            latency = latency,
            ada_scheduler_forward = ada_scheduler_forward,
        )
        # print("after vit's tranasformer: ", x.shape)                  # [num_patchs (n_tokens), bs, dim] [50, 4(64), 768]
        x = x.permute(1, 0, 2)  # LND -> NLD
        # print("self.attn_pool: ", self.attn_pool)
        # print("self.final_ln_after_pool: ", self.final_ln_after_pool)
        # assert False

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

        ### compute macs:
        model_macs = self.get_macs().to("cuda")
        # print("==== drop_block_masks====: ", drop_block_masks.shape, drop_block_masks)
        # print("model_macs: ", model_macs)
        ## always keep the first block
        if drop_block_masks is not None:
            returned_macs = (drop_block_masks * model_macs[2:]).sum(dim=-1) + model_macs[0] + model_macs[1]
        else: 
            returned_macs = None

        return pooled, returned_macs
