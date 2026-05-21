"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch16_siglip_256_infer
Pattern hash: d88c39614b6c
Shape hash: 51fb4b40
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 768], f32), T([768], f32), S([128, 1, 768]), S([128, 1, 12, 64]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[128, 768]", arg152_1: "f32[768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:102 in forward, code: q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)
        reshape_default: "f32[128, 1, 768]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        add_tensor: "f32[128, 1, 768]" = torch.ops.aten.add.Tensor(reshape_default, arg152_1);  reshape_default = arg152_1 = None
        reshape_default_1: "f32[128, 1, 12, 64]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        permute_default: "f32[128, 12, 1, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        return permute_default



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
