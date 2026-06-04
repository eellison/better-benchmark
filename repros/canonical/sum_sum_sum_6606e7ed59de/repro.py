"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train_001
Pattern hash: 6606e7ed59de
Shape hash: fa6b1d2e
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 1024], f32), T([1024], f32), T([128, 7, 7, 1024], f32), T([128, 7, 7, 1], f32), T([128, 1, 1], b8), S([128, 7, 7, 1024]), S([128, 49, 1024]), S([6272, 1024]), S([1024]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[128, 1024]", arg178_1: "f32[1024]", arg437_1: "f32[128, 7, 7, 1024]", arg439_1: "f32[128, 7, 7, 1]", arg436_1: "b8[128, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        unsqueeze_default: "f32[128, 1, 1024]" = torch.ops.aten.unsqueeze.default(mm, 1);  mm = None
        unsqueeze_default_1: "f32[128, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        expand_default: "f32[128, 7, 7, 1024]" = torch.ops.aten.expand.default(unsqueeze_default_1, _shape_param_0);  unsqueeze_default_1 = _shape_param_0 = None
        div_scalar: "f32[128, 7, 7, 1024]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        mul_tensor: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_scalar, arg178_1);  arg178_1 = None
        mul_tensor_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1024)
        sum_dim_int_list: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        mul_tensor_2: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg437_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(arg437_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(arg439_1, sub_tensor_1);  arg439_1 = sub_tensor_1 = None
        mul_tensor_5: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_scalar, arg437_1);  arg437_1 = None
        sum_dim_int_list_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1, 2]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(div_scalar, [0, 1, 2]);  div_scalar = None
        view_default: "f32[128, 49, 1024]" = torch.ops.aten.view.default(mul_tensor_4, _shape_param_1);  mul_tensor_4 = _shape_param_1 = None
        convert_element_type_default: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg436_1, torch.float32);  arg436_1 = None
        div_tensor: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.8999999985098839);  convert_element_type_default = None
        mul_tensor_6: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(view_default, div_tensor);  view_default = div_tensor = None
        view_default_1: "f32[6272, 1024]" = torch.ops.aten.view.default(mul_tensor_6, _shape_param_2);  mul_tensor_6 = _shape_param_2 = None
        permute_default: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_default_1, [1, 0])
        sum_dim_int_list_4: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_default_1, [0], True);  view_default_1 = None
        view_default_2: "f32[1024]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_3);  sum_dim_int_list_4 = _shape_param_3 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default, view_default_2)

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
