"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_train
Pattern hash: e8af10b3361f
Shape hash: 36078fca
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
_shapes_config = "(T([25216, 768], f32), T([128, 197, 768], f32), S([128, 197, 768]), S([768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_47: "f32[25216, 768]", slice_scatter: "f32[128, 197, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[128, 197, 768]" = torch.ops.aten.reshape.default(addmm_47, _shape_param_0);  addmm_47 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_tensor: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(slice_scatter, reshape_default);  slice_scatter = reshape_default = None
        sum_dim_int_list: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1], True);  mul_tensor = None
        reshape_default_1: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_1);  sum_dim_int_list = _shape_param_1 = None
        return reshape_default_1



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
