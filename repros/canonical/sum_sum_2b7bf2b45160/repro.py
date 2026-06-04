"""
Standalone repro captured via capture_hook.
Label: torchbench_mobilenet_v3_large_train_001
Pattern hash: 2b7bf2b45160
Shape hash: 5668d9fa
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([256, 960], f32), T([256, 960, 7, 7], f32), T([1, 960, 1, 1], f32), T([1, 960, 1, 1], f32), T([960], f32), T([960], f32), T([], f32), S([256, 960, 1, 1]), S([256, 960, 7, 7]))"

class Repro(torch.nn.Module):
    def forward(self, mm_2: "f32[256, 960]", arg315_1: "f32[256, 960, 7, 7]", arg316_1: "f32[1, 960, 1, 1]", arg317_1: "f32[1, 960, 1, 1]", arg130_1: "f32[960]", arg131_1: "f32[960]", full: "f32[]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[256, 960, 1, 1]" = torch.ops.aten.view.default(mm_2, _shape_param_0);  mm_2 = _shape_param_0 = None
        expand_default: "f32[256, 960, 7, 7]" = torch.ops.aten.expand.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        div_scalar: "f32[256, 960, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        sub_tensor: "f32[256, 960, 7, 7]" = torch.ops.aten.sub.Tensor(arg315_1, arg316_1)
        mul_tensor: "f32[256, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, arg317_1);  sub_tensor = None
        unsqueeze_default: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1)
        unsqueeze_default_1: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[256, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
        unsqueeze_default_3: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[256, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        le_scalar: "b8[256, 960, 7, 7]" = torch.ops.aten.le.Scalar(add_tensor, -3)
        lt_scalar: "b8[256, 960, 7, 7]" = torch.ops.aten.lt.Scalar(add_tensor, 3)
        div_tensor: "f32[256, 960, 7, 7]" = torch.ops.aten.div.Tensor(add_tensor, 3);  add_tensor = None
        add_tensor_1: "f32[256, 960, 7, 7]" = torch.ops.aten.add.Tensor(div_tensor, 0.5);  div_tensor = None
        mul_tensor_2: "f32[256, 960, 7, 7]" = torch.ops.aten.mul.Tensor(div_scalar, add_tensor_1);  add_tensor_1 = None
        where_self: "f32[256, 960, 7, 7]" = torch.ops.aten.where.self(lt_scalar, mul_tensor_2, div_scalar);  lt_scalar = mul_tensor_2 = div_scalar = None
        where_self_1: "f32[256, 960, 7, 7]" = torch.ops.aten.where.self(le_scalar, full, where_self);  le_scalar = full = where_self = None
        squeeze_dims: "f32[960]" = torch.ops.aten.squeeze.dims(arg316_1, [0, 2, 3]);  arg316_1 = None
        unsqueeze_default_4: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_1: "f32[256, 960, 7, 7]" = torch.ops.aten.sub.Tensor(arg315_1, unsqueeze_default_6);  arg315_1 = unsqueeze_default_6 = None
        mul_tensor_3: "f32[256, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_1)
        sum_dim_int_list_1: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 2, 3]);  mul_tensor_3 = None
        mul_tensor_4: "f32[960]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 7.971938775510203e-05);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_8: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_5: "f32[960]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 7.971938775510203e-05)
        squeeze_dims_1: "f32[960]" = torch.ops.aten.squeeze.dims(arg317_1, [0, 2, 3]);  arg317_1 = None
        mul_tensor_6: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_7: "f32[960]" = torch.ops.aten.mul.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None
        unsqueeze_default_10: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_11: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_8: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, arg130_1);  arg130_1 = None
        unsqueeze_default_13: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_tensor_8, 0);  mul_tensor_8 = None
        unsqueeze_default_14: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_9: "f32[256, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[256, 960, 7, 7]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_9);  where_self_1 = mul_tensor_9 = None
        sub_tensor_3: "f32[256, 960, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_10: "f32[256, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        mul_tensor_11: "f32[960]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, squeeze_dims_1);  sum_dim_int_list_1 = squeeze_dims_1 = None
        return (mul_tensor_10, mul_tensor_11)

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
