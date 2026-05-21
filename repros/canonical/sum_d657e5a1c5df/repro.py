"""
Standalone repro captured via capture_hook.
Label: torchbench_nvidia_deeprecommender_train_001
Pattern hash: d657e5a1c5df
Shape hash: d3b3c4ce
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
_shapes_config = "(T([1024, 1024], b8), T([1024, 1024], f32), T([1024, 1024], f32), S([1024]))"

class Repro(torch.nn.Module):
    def forward(self, arg11_1: "b8[1024, 1024]", mm_4: "f32[1024, 1024]", arg10_1: "f32[1024, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.float32);  arg11_1 = None
        mul_tensor: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 5.000000000000001);  convert_element_type_default = None
        mul_tensor_1: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(mm_4, mul_tensor);  mm_4 = mul_tensor = None
        le_scalar: "b8[1024, 1024]" = torch.ops.aten.le.Scalar(arg10_1, 0)
        mul_tensor_2: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 1)
        mul_tensor_3: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.7580993408473766);  mul_tensor_2 = None
        mul_tensor_4: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(arg10_1, 1);  arg10_1 = None
        exp_default: "f32[1024, 1024]" = torch.ops.aten.exp.default(mul_tensor_4);  mul_tensor_4 = None
        mul_tensor_5: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_3, exp_default);  mul_tensor_3 = exp_default = None
        mul_tensor_6: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 1.0507009873554805);  mul_tensor_1 = None
        where_self: "f32[1024, 1024]" = torch.ops.aten.where.self(le_scalar, mul_tensor_5, mul_tensor_6);  le_scalar = mul_tensor_5 = mul_tensor_6 = None
        permute_default: "f32[1024, 1024]" = torch.ops.aten.permute.default(where_self, [1, 0])
        sum_dim_int_list: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(where_self, [0], True);  where_self = None
        view_default: "f32[1024]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
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
