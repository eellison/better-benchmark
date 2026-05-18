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
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, _tensor_constant0: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:196 in _expand_mask, code: inverted_mask = torch.tensor(1.0, dtype=dtype) - expanded_mask
        lift_fresh_copy_default: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:717 in forward, code: attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)
        full_default: "f32[256, 128]" = torch.ops.aten.full.default([256, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:194 in _expand_mask, code: expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        unsqueeze_default: "f32[256, 1, 128]" = torch.ops.aten.unsqueeze.default(full_default, 1);  full_default = None
        unsqueeze_default_1: "f32[256, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        expand_default: "f32[256, 1, 128, 128]" = torch.ops.aten.expand.default(unsqueeze_default_1, [256, 1, 128, 128]);  unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:196 in _expand_mask, code: inverted_mask = torch.tensor(1.0, dtype=dtype) - expanded_mask
        sub_tensor: "f32[256, 1, 128, 128]" = torch.ops.aten.sub.Tensor(lift_fresh_copy_default, expand_default);  lift_fresh_copy_default = expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:198 in _expand_mask, code: return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
        convert_element_type_default: "b8[256, 1, 128, 128]" = torch.ops.prims.convert_element_type.default(sub_tensor, torch.bool)
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_self: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(convert_element_type_default, scalar_tensor_default, sub_tensor);  convert_element_type_default = scalar_tensor_default = sub_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:392 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_default_1: "f32[256, 12, 128, 128]" = torch.ops.aten.expand.default(where_self, [256, 12, 128, 128])
        expand_default_2: "f32[256, 12, 128, 128]" = torch.ops.aten.expand.default(where_self, [256, 12, 128, 128])
        expand_default_3: "f32[256, 12, 128, 128]" = torch.ops.aten.expand.default(where_self, [256, 12, 128, 128])
        expand_default_4: "f32[256, 12, 128, 128]" = torch.ops.aten.expand.default(where_self, [256, 12, 128, 128])
        expand_default_5: "f32[256, 12, 128, 128]" = torch.ops.aten.expand.default(where_self, [256, 12, 128, 128])
        expand_default_6: "f32[256, 12, 128, 128]" = torch.ops.aten.expand.default(where_self, [256, 12, 128, 128]);  where_self = None
        return (expand_default_1, expand_default_2, expand_default_3, expand_default_4, expand_default_5, expand_default_6)


def _default_make_inputs():
    return [
    torch.randn([], dtype=torch.float32, device='cpu'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
