"""
Standalone repro captured via capture_hook.
Label: torchbench_soft_actor_critic_train_001
Pattern hash: 7c50e5b80bfe
Shape hash: d636d51d
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([256, 1], f32), T([256, 1], f32), T([256, 1], f32), T([256, 1], f32), T([256, 1], f32), T([256, 1], f32), S([2]))"

class Repro(torch.nn.Module):
    def forward(self, arg8_1: "f32[256, 1]", arg9_1: "f32[256, 1]", arg7_1: "f32[256, 1]", arg10_1: "f32[256, 1]", arg5_1: "f32[256, 1]", arg6_1: "f32[256, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        add_tensor: "f32[256, 1]" = torch.ops.aten.add.Tensor(arg8_1, arg9_1);  arg8_1 = arg9_1 = None
        add_tensor_1: "f32[256, 1]" = torch.ops.aten.add.Tensor(arg7_1, arg10_1);  arg7_1 = arg10_1 = None
        mul_tensor: "f32[256, 1]" = torch.ops.aten.mul.Tensor(add_tensor, arg5_1);  add_tensor = arg5_1 = None
        mul_tensor_1: "f32[256, 1]" = torch.ops.aten.mul.Tensor(mul_tensor, 6.0);  mul_tensor = None
        mul_tensor_2: "f32[256, 1]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg6_1);  mul_tensor_1 = arg6_1 = None
        cat_default: "f32[256, 2]" = torch.ops.aten.cat.default([add_tensor_1, mul_tensor_2], 1);  add_tensor_1 = mul_tensor_2 = None
        permute_default: "f32[2, 256]" = torch.ops.aten.permute.default(cat_default, [1, 0])
        sum_dim_int_list: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(cat_default, [0], True);  cat_default = None
        view_default: "f32[2]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        return (permute_default, view_default)

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
