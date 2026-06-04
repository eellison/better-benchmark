"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch16_siglip_256_train
Pattern hash: 5e60905fccb8
Shape hash: bec85e89
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 768], f32), S([128, 768]), S([768]))"

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[128, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:126 in forward, code: x = x[:, 0]
        full_default: "f32[128, 1, 768]" = torch.ops.aten.full.default([128, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default: "f32[128, 1, 768]" = torch.ops.aten.select_scatter.default(full_default, tangents_1, 1, 0);  full_default = tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[128, 768]" = torch.ops.aten.reshape.default(select_scatter_default, _shape_param_0);  select_scatter_default = _shape_param_0 = None
        permute_default: "f32[768, 128]" = torch.ops.aten.permute.default(reshape_default, [1, 0])
        sum_dim_int_list: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0], True);  reshape_default = None
        reshape_default_1: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_1);  sum_dim_int_list = _shape_param_1 = None
        return (permute_default, reshape_default_1)

def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)

def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
