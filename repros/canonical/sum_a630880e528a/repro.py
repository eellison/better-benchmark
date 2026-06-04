"""
Standalone repro captured via capture_hook.
Label: torchbench_LearningToPaint_train_001
Pattern hash: a630880e528a
Shape hash: 846f124a
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([1024, 65], f32), T([1024, 65], f32), S([65]))"

class Repro(torch.nn.Module):
    def forward(self, arg104_1: "f32[1024, 65]", arg126_1: "f32[1024, 65]", _shape_param_0):
        # No stacktrace found for following nodes
        sub_tensor: "f32[1024, 65]" = torch.ops.aten.sub.Tensor(1, arg104_1)
        mul_tensor: "f32[1024, 65]" = torch.ops.aten.mul.Tensor(arg104_1, sub_tensor);  arg104_1 = sub_tensor = None
        mul_tensor_1: "f32[1024, 65]" = torch.ops.aten.mul.Tensor(arg126_1, mul_tensor);  arg126_1 = mul_tensor = None
        permute_default: "f32[65, 1024]" = torch.ops.aten.permute.default(mul_tensor_1, [1, 0])
        sum_dim_int_list: "f32[1, 65]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0], True);  mul_tensor_1 = None
        view_default: "f32[65]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
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
