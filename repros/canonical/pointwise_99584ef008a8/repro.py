"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:165 in _make_causal_mask, code: mask_cond = torch.arange(mask.size(-1), device=device)
        iota_default: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:166 in _make_causal_mask, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        add_tensor: "i64[256]" = torch.ops.aten.add.Tensor(iota_default, 1)
        reshape_default: "i64[256, 1]" = torch.ops.aten.reshape.default(add_tensor, [256, 1]);  add_tensor = None
        lt_tensor: "b8[256, 256]" = torch.ops.aten.lt.Tensor(iota_default, reshape_default);  iota_default = reshape_default = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:164 in _make_causal_mask, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        full_default_1: "f32[256, 256]" = torch.ops.aten.full.default([256, 256], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:166 in _make_causal_mask, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        where_self: "f32[256, 256]" = torch.ops.aten.where.self(lt_tensor, full_default, full_default_1);  lt_tensor = full_default = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:184 in _make_causal_mask, code: return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
        unsqueeze_default: "f32[1, 256, 256]" = torch.ops.aten.unsqueeze.default(where_self, 0);  where_self = None
        unsqueeze_default_1: "f32[1, 1, 256, 256]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None
        expand_default: "f32[64, 1, 256, 256]" = torch.ops.aten.expand.default(unsqueeze_default_1, [64, 1, 256, 256]);  unsqueeze_default_1 = None

        # No stacktrace found for following nodes
        expand_default_1: "f32[64, 16, 256, 256]" = torch.ops.aten.expand.default(expand_default, [64, 16, 256, 256])
        expand_default_2: "f32[64, 16, 256, 256]" = torch.ops.aten.expand.default(expand_default, [64, 16, 256, 256])
        expand_default_3: "f32[64, 16, 256, 256]" = torch.ops.aten.expand.default(expand_default, [64, 16, 256, 256])
        expand_default_4: "f32[64, 16, 256, 256]" = torch.ops.aten.expand.default(expand_default, [64, 16, 256, 256])
        expand_default_5: "f32[64, 16, 256, 256]" = torch.ops.aten.expand.default(expand_default, [64, 16, 256, 256])
        expand_default_6: "f32[64, 16, 256, 256]" = torch.ops.aten.expand.default(expand_default, [64, 16, 256, 256])
        expand_default_7: "f32[64, 16, 256, 256]" = torch.ops.aten.expand.default(expand_default, [64, 16, 256, 256])
        expand_default_8: "f32[64, 16, 256, 256]" = torch.ops.aten.expand.default(expand_default, [64, 16, 256, 256])
        expand_default_9: "f32[64, 16, 256, 256]" = torch.ops.aten.expand.default(expand_default, [64, 16, 256, 256])
        expand_default_10: "f32[64, 16, 256, 256]" = torch.ops.aten.expand.default(expand_default, [64, 16, 256, 256])
        expand_default_11: "f32[64, 16, 256, 256]" = torch.ops.aten.expand.default(expand_default, [64, 16, 256, 256])
        expand_default_12: "f32[64, 16, 256, 256]" = torch.ops.aten.expand.default(expand_default, [64, 16, 256, 256]);  expand_default = None
        return (expand_default_1, expand_default_2, expand_default_3, expand_default_4, expand_default_5, expand_default_6, expand_default_7, expand_default_8, expand_default_9, expand_default_10, expand_default_11, expand_default_12)


def _default_make_inputs():
    return [

    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
