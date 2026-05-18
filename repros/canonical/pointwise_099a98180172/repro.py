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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:807 in forward, code: attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)
        full_default: "f32[32, 512]" = torch.ops.aten.full.default([32, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:194 in _expand_mask, code: expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        unsqueeze_default: "f32[32, 1, 512]" = torch.ops.aten.unsqueeze.default(full_default, 1);  full_default = None
        unsqueeze_default_1: "f32[32, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        expand_default: "f32[32, 1, 512, 512]" = torch.ops.aten.expand.default(unsqueeze_default_1, [32, 1, 512, 512]);  unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:196 in _expand_mask, code: inverted_mask = torch.tensor(1.0, dtype=dtype) - expanded_mask
        sub_tensor: "f32[32, 1, 512, 512]" = torch.ops.aten.sub.Tensor(lift_fresh_copy_default, expand_default);  lift_fresh_copy_default = expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:198 in _expand_mask, code: return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
        convert_element_type_default: "b8[32, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_tensor, torch.bool)
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_self: "f32[32, 1, 512, 512]" = torch.ops.aten.where.self(convert_element_type_default, scalar_tensor_default, sub_tensor);  convert_element_type_default = scalar_tensor_default = sub_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:363 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_default_1: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_self, [32, 12, 512, 512])
        expand_default_2: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_self, [32, 12, 512, 512])
        expand_default_3: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_self, [32, 12, 512, 512])
        expand_default_4: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_self, [32, 12, 512, 512])
        expand_default_5: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_self, [32, 12, 512, 512])
        expand_default_6: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_self, [32, 12, 512, 512])
        expand_default_7: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_self, [32, 12, 512, 512])
        expand_default_8: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_self, [32, 12, 512, 512])
        expand_default_9: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_self, [32, 12, 512, 512])
        expand_default_10: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_self, [32, 12, 512, 512])
        expand_default_11: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_self, [32, 12, 512, 512])
        expand_default_12: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_self, [32, 12, 512, 512]);  where_self = None
        return (expand_default_1, expand_default_2, expand_default_3, expand_default_4, expand_default_5, expand_default_6, expand_default_7, expand_default_8, expand_default_9, expand_default_10, expand_default_11, expand_default_12)


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
