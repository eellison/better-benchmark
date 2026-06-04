"""
Standalone repro captured via capture_hook.
Label: torchbench_lennard_jones_train_001
Pattern hash: 27f326f22f76
Shape hash: e85fcfc3
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([1, 16], f32), T([1000, 1], f32), T([1000, 16], f32), S([16]))"

class Repro(torch.nn.Module):
    def forward(self, arg4_1: "f32[1, 16]", arg9_1: "f32[1000, 1]", arg8_1: "f32[1000, 16]", _shape_param_0):
        # No stacktrace found for following nodes
        permute_default: "f32[16, 1]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        permute_default_1: "f32[1, 16]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        mul_tensor: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(arg9_1, permute_default_1);  arg9_1 = permute_default_1 = None
        mul_tensor_1: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(arg8_1, arg8_1);  arg8_1 = None
        sub_tensor: "f32[1000, 16]" = torch.ops.aten.sub.Tensor(1, mul_tensor_1);  mul_tensor_1 = None
        mul_tensor_2: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(mul_tensor, sub_tensor);  mul_tensor = sub_tensor = None
        permute_default_2: "f32[16, 1000]" = torch.ops.aten.permute.default(mul_tensor_2, [1, 0])
        sum_dim_int_list: "f32[1, 16]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0], True);  mul_tensor_2 = None
        view_default: "f32[16]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        return (permute_default_2, view_default)

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
