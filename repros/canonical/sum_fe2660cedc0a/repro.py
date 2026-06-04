"""
Standalone repro captured via capture_hook.
Label: hf_GoogleFnet_train_001
Pattern hash: fe2660cedc0a
Shape hash: 95bab77d
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([32, 512, 768, 2], f32), T([32, 512, 768], f32), T([32, 512, 768], b8), S([16384, 768]), S([768]))"

class Repro(torch.nn.Module):
    def forward(self, view_as_real_11: "f32[32, 512, 768, 2]", mul_332: "f32[32, 512, 768]", arg59_1: "b8[32, 512, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        select_int: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_11, 3, 0);  view_as_real_11 = None
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_332, select_int);  mul_332 = select_int = None
        convert_element_type_default: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg59_1, torch.float32);  arg59_1 = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor);  add_tensor = mul_tensor = None
        view_default: "f32[16384, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_0);  mul_tensor_1 = _shape_param_0 = None
        permute_default: "f32[768, 16384]" = torch.ops.aten.permute.default(view_default, [1, 0])
        sum_dim_int_list: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_default, [0], True);  view_default = None
        view_default_1: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_1);  sum_dim_int_list = _shape_param_1 = None
        return (permute_default, view_default_1)

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
