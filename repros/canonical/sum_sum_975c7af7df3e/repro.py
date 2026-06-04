"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_resnest_train_001
Pattern hash: 975c7af7df3e
Shape hash: dfc2f092
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([32, 2, 1, 64], f32), T([32, 2, 64, 56, 56], f32, stride=(200704, 0, 3136, 56, 1)), T([32, 64, 1, 1], f32), T([32, 128, 56, 56], f32), T([], f32), T([1, 128, 1, 1], f32), T([32, 128, 56, 56], f32), T([1, 128, 1, 1], f32), T([128], f32), S([32, -1]), S([32, -1, 1, 1]), S([32, 2, 64, 1, 1]), S([32, 64, 56, 56]), S([32, 2, 64, 56, 56]), S([32, 128, 56, 56]))"

class Repro(torch.nn.Module):
    def forward(self, div_7: "f32[32, 2, 1, 64]", expand_10: "f32[32, 2, 64, 56, 56]", getitem_63: "f32[32, 64, 1, 1]", relu_4: "f32[32, 128, 56, 56]", full: "f32[]", arg74_1: "f32[1, 128, 1, 1]", arg73_1: "f32[32, 128, 56, 56]", arg75_1: "f32[1, 128, 1, 1]", arg11_1: "f32[128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view_default: "f32[32, 128]" = torch.ops.aten.view.default(div_7, _shape_param_0);  div_7 = _shape_param_0 = None
        view_default_1: "f32[32, 128, 1, 1]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        view_default_2: "f32[32, 2, 64, 1, 1]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        mul_tensor: "f32[32, 2, 64, 56, 56]" = torch.ops.aten.mul.Tensor(expand_10, view_default_2);  expand_10 = view_default_2 = None
        expand_default: "f32[32, 64, 56, 56]" = torch.ops.aten.expand.default(getitem_63, _shape_param_3);  getitem_63 = _shape_param_3 = None
        div_scalar: "f32[32, 64, 56, 56]" = torch.ops.aten.div.Scalar(expand_default, 3136);  expand_default = None
        unsqueeze_default: "f32[32, 1, 64, 56, 56]" = torch.ops.aten.unsqueeze.default(div_scalar, 1);  div_scalar = None
        expand_default_1: "f32[32, 2, 64, 56, 56]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_4);  unsqueeze_default = _shape_param_4 = None
        add_tensor: "f32[32, 2, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor, expand_default_1);  mul_tensor = expand_default_1 = None
        view_default_3: "f32[32, 128, 56, 56]" = torch.ops.aten.view.default(add_tensor, _shape_param_5);  add_tensor = _shape_param_5 = None
        le_scalar: "b8[32, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_self: "f32[32, 128, 56, 56]" = torch.ops.aten.where.self(le_scalar, full, view_default_3);  le_scalar = full = view_default_3 = None
        squeeze_dims: "f32[128]" = torch.ops.aten.squeeze.dims(arg74_1, [0, 2, 3]);  arg74_1 = None
        unsqueeze_default_1: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_2: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 2);  unsqueeze_default_1 = None
        unsqueeze_default_3: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 3);  unsqueeze_default_2 = None
        sum_dim_int_list: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(arg73_1, unsqueeze_default_3);  arg73_1 = unsqueeze_default_3 = None
        mul_tensor_1: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 2, 3]);  mul_tensor_1 = None
        mul_tensor_2: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 9.964923469387754e-06);  sum_dim_int_list = None
        unsqueeze_default_4: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_2, 0);  mul_tensor_2 = None
        unsqueeze_default_5: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        mul_tensor_3: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 9.964923469387754e-06)
        squeeze_dims_1: "f32[128]" = torch.ops.aten.squeeze.dims(arg75_1, [0, 2, 3]);  arg75_1 = None
        mul_tensor_4: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_5: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None
        unsqueeze_default_7: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_8: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_6: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, arg11_1);  arg11_1 = None
        unsqueeze_default_10: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_9);  sub_tensor = unsqueeze_default_9 = None
        sub_tensor_1: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_7);  where_self = mul_tensor_7 = None
        sub_tensor_2: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_6);  sub_tensor_1 = unsqueeze_default_6 = None
        mul_tensor_8: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_12);  sub_tensor_2 = unsqueeze_default_12 = None
        mul_tensor_9: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, squeeze_dims_1);  sum_dim_int_list_1 = squeeze_dims_1 = None
        return (mul_tensor_8, mul_tensor_9)

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
