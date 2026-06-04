"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_resnest_infer_000
Pattern hash: b9c8bc430873
Shape hash: 7c0eb9de
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([32, 128, 1, 1], f16), T([32, 2, 64, 56, 56], f16), S([32, 1, 2, -1]), S([32, -1]), S([32, -1, 1, 1]), S([32, 2, 64, 1, 1]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_6: "f16[32, 128, 1, 1]", view: "f16[32, 2, 64, 56, 56]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f16[32, 1, 2, 64]" = torch.ops.aten.view.default(convolution_6, _shape_param_0);  convolution_6 = _shape_param_0 = None
        permute_default: "f16[32, 2, 1, 64]" = torch.ops.aten.permute.default(view_default, [0, 2, 1, 3]);  view_default = None
        convert_element_type_default: "f32[32, 2, 1, 64]" = torch.ops.prims.convert_element_type.default(permute_default, torch.float32);  permute_default = None
        amax_default: "f32[32, 1, 1, 64]" = torch.ops.aten.amax.default(convert_element_type_default, [1], True)
        sub_tensor: "f32[32, 2, 1, 64]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default);  convert_element_type_default = amax_default = None
        exp_default: "f32[32, 2, 1, 64]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[32, 1, 1, 64]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True)
        div_tensor: "f32[32, 2, 1, 64]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_1: "f16[32, 2, 1, 64]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float16);  div_tensor = None
        view_default_1: "f16[32, 128]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_1);  convert_element_type_default_1 = _shape_param_1 = None
        view_default_2: "f16[32, 128, 1, 1]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        view_default_3: "f16[32, 2, 64, 1, 1]" = torch.ops.aten.view.default(view_default_2, _shape_param_3);  view_default_2 = _shape_param_3 = None
        mul_tensor: "f16[32, 2, 64, 56, 56]" = torch.ops.aten.mul.Tensor(view, view_default_3);  view = view_default_3 = None
        sum_dim_int_list_1: "f16[32, 64, 56, 56]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1]);  mul_tensor = None
        return sum_dim_int_list_1

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
