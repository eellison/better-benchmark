"""
Standalone repro captured via capture_hook.
Label: hf_DistillGPT2_inference
Pattern hash: f516f32a662d
Shape hash: d475d2cb
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_57: "f32[8, 512, 768]", _shape_param_0, getitem_58: "f32[8, 512, 768]", _shape_param_1, getitem_59: "f32[8, 512, 768]", _shape_param_2, expand_1: "b8[8, 1, 512, 512]", _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        reshape_default: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_57, _shape_param_0);  getitem_57 = _shape_param_0 = None
        permute_default: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        reshape_default_1: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_58, _shape_param_1);  getitem_58 = _shape_param_1 = None
        permute_default_1: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        reshape_default_2: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_59, _shape_param_2);  getitem_59 = _shape_param_2 = None
        permute_default_2: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default, full_default_1);  expand_1 = full_default = full_default_1 = None
        expand_default: "f32[8, 12, 512, 512]" = torch.ops.aten.expand.default(where_self, _shape_param_3);  where_self = _shape_param_3 = None
        return (permute_default, permute_default_1, permute_default_2, expand_default)


def _default_make_inputs():
    return [
    torch.randn(9435648, dtype=torch.float32, device='cuda').as_strided([8, 512, 768], [1179648, 2304, 1]),  # getitem_57
    [8, 512, -1, 64],  # _shape_param_0
    torch.randn(9435648, dtype=torch.float32, device='cuda').as_strided([8, 512, 768], [1179648, 2304, 1]),  # getitem_58
    [8, 512, -1, 64],  # _shape_param_1
    torch.randn(9435648, dtype=torch.float32, device='cuda').as_strided([8, 512, 768], [1179648, 2304, 1]),  # getitem_59
    [8, 512, -1, 64],  # _shape_param_2
    torch.randint(0, 2, [8, 1, 512, 512], dtype=torch.bool, device='cuda'),
    [8, 12, 512, 512],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
