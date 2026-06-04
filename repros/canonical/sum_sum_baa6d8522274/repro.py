"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train_001
Pattern hash: baa6d8522274
Shape hash: 3f7fd6ec
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 512, 13, 13], b8), T([512, 512, 13, 13], f32), T([512, 256, 13, 13], b8), T([], f32), T([512, 256, 13, 13], b8))"

class Repro(torch.nn.Module):
    def forward(self, arg46_1: "b8[512, 512, 13, 13]", getitem: "f32[512, 512, 13, 13]", arg49_1: "b8[512, 256, 13, 13]", full: "f32[]", arg50_1: "b8[512, 256, 13, 13]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[512, 512, 13, 13]" = torch.ops.prims.convert_element_type.default(arg46_1, torch.float32);  arg46_1 = None
        mul_tensor: "f32[512, 512, 13, 13]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 2.0);  convert_element_type_default = None
        mul_tensor_1: "f32[512, 512, 13, 13]" = torch.ops.aten.mul.Tensor(getitem, mul_tensor);  getitem = mul_tensor = None
        slice_tensor: "f32[512, 256, 13, 13]" = torch.ops.aten.slice.Tensor(mul_tensor_1, 1, 0, 256)
        slice_tensor_1: "f32[512, 256, 13, 13]" = torch.ops.aten.slice.Tensor(mul_tensor_1, 1, 256, 512);  mul_tensor_1 = None
        where_self: "f32[512, 256, 13, 13]" = torch.ops.aten.where.self(arg49_1, full, slice_tensor_1);  arg49_1 = slice_tensor_1 = None
        sum_dim_int_list: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3]);  where_self = None
        where_self_1: "f32[512, 256, 13, 13]" = torch.ops.aten.where.self(arg50_1, full, slice_tensor);  arg50_1 = full = slice_tensor = None
        sum_dim_int_list_1: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3]);  where_self_1 = None
        return (sum_dim_int_list, sum_dim_int_list_1)

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
