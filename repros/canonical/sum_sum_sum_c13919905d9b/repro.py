"""
Standalone repro captured via capture_hook.
Label: timm_repvgg_a2_train_001
Pattern hash: c13919905d9b
Shape hash: 8282b150
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 1408], f32), T([128, 1408, 7, 7], f32), T([1, 1408, 1, 1], f32), T([1, 1408, 1, 1], f32), T([1408], f32), T([1408], f32), T([128, 1408, 7, 7], f32), T([1, 1408, 1, 1], f32), T([1, 1408, 1, 1], f32), T([1408], f32), T([1408], f32), S([128, 1408, 1, 1]), S([128, 1408, 7, 7]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[128, 1408]", arg231_1: "f32[128, 1408, 7, 7]", arg232_1: "f32[1, 1408, 1, 1]", arg233_1: "f32[1, 1408, 1, 1]", arg103_1: "f32[1408]", arg104_1: "f32[1408]", arg234_1: "f32[128, 1408, 7, 7]", arg235_1: "f32[1, 1408, 1, 1]", arg236_1: "f32[1, 1408, 1, 1]", arg106_1: "f32[1408]", arg107_1: "f32[1408]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[128, 1408, 1, 1]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        squeeze_dim: "f32[128, 1408, 1]" = torch.ops.aten.squeeze.dim(view_default, 3);  view_default = None
        squeeze_dim_1: "f32[128, 1408]" = torch.ops.aten.squeeze.dim(squeeze_dim, 2);  squeeze_dim = None
        full_default: "f32[180224]" = torch.ops.aten.full.default([180224], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter_default: "f32[180224]" = torch.ops.aten.as_strided_scatter.default(full_default, squeeze_dim_1, [128, 1408], [1408, 1], 0);  full_default = squeeze_dim_1 = None
        as_strided_default: "f32[128, 1408, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter_default, [128, 1408, 1, 1], [1408, 1, 1, 1], 0);  as_strided_scatter_default = None
        expand_default: "f32[128, 1408, 7, 7]" = torch.ops.aten.expand.default(as_strided_default, _shape_param_1);  as_strided_default = _shape_param_1 = None
        div_scalar: "f32[128, 1408, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        sub_tensor: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(arg231_1, arg232_1)
        mul_tensor: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, arg233_1);  sub_tensor = None
        unsqueeze_default: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg103_1, -1)
        unsqueeze_default_1: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_default_3: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        sub_tensor_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(arg234_1, arg235_1)
        mul_tensor_2: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, arg236_1);  sub_tensor_1 = None
        unsqueeze_default_4: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, -1)
        unsqueeze_default_5: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_default_7: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None
        add_tensor_2: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor, add_tensor_1);  add_tensor = add_tensor_1 = None
        relu_default: "f32[128, 1408, 7, 7]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None
        le_scalar: "b8[128, 1408, 7, 7]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[128, 1408, 7, 7]" = torch.ops.aten.where.self(le_scalar, full_default_1, div_scalar);  le_scalar = full_default_1 = div_scalar = None
        squeeze_dims: "f32[1408]" = torch.ops.aten.squeeze.dims(arg235_1, [0, 2, 3]);  arg235_1 = None
        unsqueeze_default_8: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_9: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 2);  unsqueeze_default_8 = None
        unsqueeze_default_10: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 3);  unsqueeze_default_9 = None
        sum_dim_int_list: "f32[1408]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_2: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(arg234_1, unsqueeze_default_10);  arg234_1 = unsqueeze_default_10 = None
        mul_tensor_4: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_2)
        sum_dim_int_list_1: "f32[1408]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 2, 3]);  mul_tensor_4 = None
        mul_tensor_5: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00015943877551020407);  sum_dim_int_list = None
        unsqueeze_default_11: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_12: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 2);  unsqueeze_default_11 = None
        unsqueeze_default_13: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 3);  unsqueeze_default_12 = None
        mul_tensor_6: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00015943877551020407)
        squeeze_dims_1: "f32[1408]" = torch.ops.aten.squeeze.dims(arg236_1, [0, 2, 3]);  arg236_1 = None
        mul_tensor_7: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_8: "f32[1408]" = torch.ops.aten.mul.Tensor(mul_tensor_6, mul_tensor_7);  mul_tensor_6 = mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_tensor_8, 0);  mul_tensor_8 = None
        unsqueeze_default_15: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 2);  unsqueeze_default_14 = None
        unsqueeze_default_16: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 3);  unsqueeze_default_15 = None
        mul_tensor_9: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, arg106_1);  arg106_1 = None
        unsqueeze_default_17: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_18: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 2);  unsqueeze_default_17 = None
        unsqueeze_default_19: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, 3);  unsqueeze_default_18 = None
        mul_tensor_10: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_16);  sub_tensor_2 = unsqueeze_default_16 = None
        sub_tensor_3: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_10);  mul_tensor_10 = None
        sub_tensor_4: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_3, unsqueeze_default_13);  sub_tensor_3 = unsqueeze_default_13 = None
        mul_tensor_11: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_19);  sub_tensor_4 = unsqueeze_default_19 = None
        mul_tensor_12: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, squeeze_dims_1);  sum_dim_int_list_1 = squeeze_dims_1 = None
        squeeze_dims_2: "f32[1408]" = torch.ops.aten.squeeze.dims(arg232_1, [0, 2, 3]);  arg232_1 = None
        unsqueeze_default_20: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(squeeze_dims_2, 0);  squeeze_dims_2 = None
        unsqueeze_default_21: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 2);  unsqueeze_default_20 = None
        unsqueeze_default_22: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_21, 3);  unsqueeze_default_21 = None
        sum_dim_int_list_2: "f32[1408]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_5: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(arg231_1, unsqueeze_default_22);  arg231_1 = unsqueeze_default_22 = None
        mul_tensor_13: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_5)
        sum_dim_int_list_3: "f32[1408]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 2, 3]);  mul_tensor_13 = None
        mul_tensor_14: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 0.00015943877551020407);  sum_dim_int_list_2 = None
        unsqueeze_default_23: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_tensor_14, 0);  mul_tensor_14 = None
        unsqueeze_default_24: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 2);  unsqueeze_default_23 = None
        unsqueeze_default_25: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, 3);  unsqueeze_default_24 = None
        mul_tensor_15: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 0.00015943877551020407)
        squeeze_dims_3: "f32[1408]" = torch.ops.aten.squeeze.dims(arg233_1, [0, 2, 3]);  arg233_1 = None
        mul_tensor_16: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, squeeze_dims_3)
        mul_tensor_17: "f32[1408]" = torch.ops.aten.mul.Tensor(mul_tensor_15, mul_tensor_16);  mul_tensor_15 = mul_tensor_16 = None
        unsqueeze_default_26: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_tensor_17, 0);  mul_tensor_17 = None
        unsqueeze_default_27: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 2);  unsqueeze_default_26 = None
        unsqueeze_default_28: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_27, 3);  unsqueeze_default_27 = None
        mul_tensor_18: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, arg103_1);  arg103_1 = None
        unsqueeze_default_29: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_tensor_18, 0);  mul_tensor_18 = None
        unsqueeze_default_30: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 2);  unsqueeze_default_29 = None
        unsqueeze_default_31: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, 3);  unsqueeze_default_30 = None
        mul_tensor_19: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_28);  sub_tensor_5 = unsqueeze_default_28 = None
        sub_tensor_6: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_19);  where_self = mul_tensor_19 = None
        sub_tensor_7: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_6, unsqueeze_default_25);  sub_tensor_6 = unsqueeze_default_25 = None
        mul_tensor_20: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_7, unsqueeze_default_31);  sub_tensor_7 = unsqueeze_default_31 = None
        mul_tensor_21: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, squeeze_dims_3);  sum_dim_int_list_3 = squeeze_dims_3 = None
        return (mul_tensor_11, mul_tensor_12, mul_tensor_20, mul_tensor_21)

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
