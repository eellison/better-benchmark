"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=amax, ranges=['512', '128', '1'], reduction_ranges=[]
#   origins: ['aten.amax.default']
#   type=sum, ranges=['512', '128', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm: "f32[512, 128, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:212 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        reshape_default: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm, [32, 16, 128, 128]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:165 in _make_causal_mask, code: mask_cond = torch.arange(mask.size(-1), device=device)
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:166 in _make_causal_mask, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        add_tensor: "i64[128]" = torch.ops.aten.add.Tensor(iota_default, 1)
        reshape_default_1: "i64[128, 1]" = torch.ops.aten.reshape.default(add_tensor, [128, 1]);  add_tensor = None
        lt_tensor: "b8[128, 128]" = torch.ops.aten.lt.Tensor(iota_default, reshape_default_1);  iota_default = reshape_default_1 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:164 in _make_causal_mask, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        full_default_1: "f32[128, 128]" = torch.ops.aten.full.default([128, 128], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:166 in _make_causal_mask, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        where_self: "f32[128, 128]" = torch.ops.aten.where.self(lt_tensor, full_default, full_default_1);  lt_tensor = full_default = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:184 in _make_causal_mask, code: return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
        unsqueeze_default: "f32[1, 128, 128]" = torch.ops.aten.unsqueeze.default(where_self, 0);  where_self = None
        unsqueeze_default_1: "f32[1, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None
        expand_default: "f32[32, 1, 128, 128]" = torch.ops.aten.expand.default(unsqueeze_default_1, [32, 1, 128, 128]);  unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:212 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        add_tensor_1: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default, expand_default);  reshape_default = expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:214 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:213 in forward, code: attn_weights = torch.max(
        maximum_default: "f32[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_tensor_1, full_default_2);  add_tensor_1 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:216 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        reshape_default_2: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(maximum_default, [512, 128, 128]);  maximum_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:222 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_default: "f32[512, 128, 1]" = torch.ops.aten.amax.default(reshape_default_2, [-1], True)
        sub_tensor: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(reshape_default_2, amax_default);  reshape_default_2 = amax_default = None
        exp_default: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        return div_tensor


def _default_make_inputs():
    return [
    torch.randn([512, 128, 128], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
