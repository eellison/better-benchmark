"""
Standalone repro captured via capture_hook.
Label: torchbench_mobilenet_v3_large_train_001
Pattern hash: 6080543764bd
Shape hash: beb38d20
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([256, 1280], b8), T([256, 1280], f32), T([256, 1280], f32), S([1280]))"

class Repro(torch.nn.Module):
    def forward(self, arg320_1: "b8[256, 1280]", mm: "f32[256, 1280]", arg319_1: "f32[256, 1280]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[256, 1280]" = torch.ops.prims.convert_element_type.default(arg320_1, torch.float32);  arg320_1 = None
        div_scalar: "f32[256, 1280]" = torch.ops.aten.div.Scalar(convert_element_type_default, 0.8);  convert_element_type_default = None
        mul_tensor: "f32[256, 1280]" = torch.ops.aten.mul.Tensor(mm, div_scalar);  mm = div_scalar = None
        le_scalar: "b8[256, 1280]" = torch.ops.aten.le.Scalar(arg319_1, -3)
        lt_scalar: "b8[256, 1280]" = torch.ops.aten.lt.Scalar(arg319_1, 3)
        div_tensor: "f32[256, 1280]" = torch.ops.aten.div.Tensor(arg319_1, 3);  arg319_1 = None
        add_tensor: "f32[256, 1280]" = torch.ops.aten.add.Tensor(div_tensor, 0.5);  div_tensor = None
        mul_tensor_1: "f32[256, 1280]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  add_tensor = None
        where_self: "f32[256, 1280]" = torch.ops.aten.where.self(lt_scalar, mul_tensor_1, mul_tensor);  lt_scalar = mul_tensor_1 = mul_tensor = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[256, 1280]" = torch.ops.aten.where.self(le_scalar, full_default, where_self);  le_scalar = full_default = where_self = None
        permute_default: "f32[1280, 256]" = torch.ops.aten.permute.default(where_self_1, [1, 0])
        sum_dim_int_list: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0], True);  where_self_1 = None
        view_default: "f32[1280]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
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
