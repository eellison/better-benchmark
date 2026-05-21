"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train_001
Pattern hash: ee85624361a0
Shape hash: e4e4a480
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
_shapes_config = "(T([512, 72, 1, 1], f32), T([512, 72, 28, 28], f32), T([512, 72, 1, 1], f32), T([512, 72, 28, 28], f32), T([], f32), T([1, 72, 1, 1], f32), T([512, 72, 28, 28], f32), T([1, 72, 1, 1], f32), T([72], f32), S([512, 72, 28, 28]))"

class Repro(torch.nn.Module):
    def forward(self, arg170_1: "f32[512, 72, 1, 1]", getitem_147: "f32[512, 72, 28, 28]", getitem_153: "f32[512, 72, 1, 1]", relu_2: "f32[512, 72, 28, 28]", full: "f32[]", arg166_1: "f32[1, 72, 1, 1]", arg165_1: "f32[512, 72, 28, 28]", arg167_1: "f32[1, 72, 1, 1]", arg23_1: "f32[72]", _shape_param_0):
        # No stacktrace found for following nodes
        add_tensor: "f32[512, 72, 1, 1]" = torch.ops.aten.add.Tensor(arg170_1, 3);  arg170_1 = None
        clamp_min_default: "f32[512, 72, 1, 1]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[512, 72, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        div_tensor: "f32[512, 72, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_default, 6);  clamp_max_default = None
        mul_tensor: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_147, div_tensor);  getitem_147 = div_tensor = None
        expand_default: "f32[512, 72, 28, 28]" = torch.ops.aten.expand.default(getitem_153, _shape_param_0);  getitem_153 = _shape_param_0 = None
        div_scalar: "f32[512, 72, 28, 28]" = torch.ops.aten.div.Scalar(expand_default, 784);  expand_default = None
        add_tensor_1: "f32[512, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor, div_scalar);  mul_tensor = div_scalar = None
        le_scalar: "b8[512, 72, 28, 28]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_self: "f32[512, 72, 28, 28]" = torch.ops.aten.where.self(le_scalar, full, add_tensor_1);  le_scalar = full = add_tensor_1 = None
        squeeze_dims: "f32[72]" = torch.ops.aten.squeeze.dims(arg166_1, [0, 2, 3]);  arg166_1 = None
        unsqueeze_default: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_1: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        sum_dim_int_list: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[512, 72, 28, 28]" = torch.ops.aten.sub.Tensor(arg165_1, unsqueeze_default_2);  arg165_1 = unsqueeze_default_2 = None
        mul_tensor_1: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 2, 3]);  mul_tensor_1 = None
        mul_tensor_2: "f32[72]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.4912308673469386e-06);  sum_dim_int_list = None
        unsqueeze_default_3: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_tensor_2, 0);  mul_tensor_2 = None
        unsqueeze_default_4: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[72]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.4912308673469386e-06)
        squeeze_dims_1: "f32[72]" = torch.ops.aten.squeeze.dims(arg167_1, [0, 2, 3]);  arg167_1 = None
        mul_tensor_4: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_5: "f32[72]" = torch.ops.aten.mul.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None
        unsqueeze_default_6: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, arg23_1);  arg23_1 = None
        unsqueeze_default_9: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_10: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_7: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_8);  sub_tensor = unsqueeze_default_8 = None
        sub_tensor_1: "f32[512, 72, 28, 28]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_7);  where_self = mul_tensor_7 = None
        sub_tensor_2: "f32[512, 72, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_5);  sub_tensor_1 = unsqueeze_default_5 = None
        mul_tensor_8: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_11);  sub_tensor_2 = unsqueeze_default_11 = None
        mul_tensor_9: "f32[72]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, squeeze_dims_1);  sum_dim_int_list_1 = squeeze_dims_1 = None
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
