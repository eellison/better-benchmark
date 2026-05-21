"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_train
Pattern hash: 7870df03ff6e
Shape hash: 44051703
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
_shapes_config = "(T([128, 197, 768], f32), T([128, 197, 768], f32), S([768]))"

class Repro(torch.nn.Module):
    def forward(self, add_132: "f32[128, 197, 768]", view_8: "f32[128, 197, 768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_tensor: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(add_132, view_8);  add_132 = view_8 = None
        sum_dim_int_list: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1], True);  mul_tensor = None
        reshape_default: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        return reshape_default



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
