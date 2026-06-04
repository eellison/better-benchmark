"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch16_siglip_256_train
Pattern hash: acbec27412d2
Shape hash: 9c7353f8
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 12, 1, 64], f32), S([128, 1, 768]), S([768]), S([128, 768]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_142: "f32[128, 12, 1, 64]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:102 in forward, code: q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)
        permute_default: "f32[128, 1, 12, 64]" = torch.ops.aten.permute.default(getitem_142, [0, 2, 1, 3]);  getitem_142 = None
        reshape_default: "f32[128, 1, 768]" = torch.ops.aten.reshape.default(permute_default, _shape_param_0);  permute_default = _shape_param_0 = None
        sum_dim_int_list: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0, 1], True)
        reshape_default_1: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_1);  sum_dim_int_list = _shape_param_1 = None
        reshape_default_2: "f32[128, 768]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_2);  reshape_default = _shape_param_2 = None
        permute_default_1: "f32[768, 128]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0]);  reshape_default_2 = None
        return (reshape_default_1, permute_default_1)

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
