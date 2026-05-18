"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s7_g53
Pattern hash: cd4360587211
Shape hash: b5b1bb4d
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm: "f16[128, 1408]", arg229_1: "f16[128, 1408, 7, 7]", arg230_1: "f32[1, 1408, 1, 1]", arg231_1: "f32[1, 1408, 1, 1]", arg59_1: "f32[1408]", arg60_1: "f32[1408]", arg233_1: "f16[128, 1408, 7, 7]", arg234_1: "f32[1, 1408, 1, 1]", arg235_1: "f32[1, 1408, 1, 1]", arg61_1: "f32[1408]", arg62_1: "f32[1408]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[128, 1408, 1, 1]" = torch.ops.aten.reshape.default(mm, [128, 1408, 1, 1]);  mm = None
        expand_default: "f16[128, 1408, 7, 7]" = torch.ops.aten.expand.default(reshape_default, [128, 1408, 7, 7]);  reshape_default = None
        div_scalar: "f16[128, 1408, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        sub_tensor: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(arg229_1, arg230_1)
        mul_tensor: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, arg231_1);  sub_tensor = None
        unsqueeze_default: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1)
        unsqueeze_default_1: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_default_3: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        convert_element_type_default: "f16[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float16);  add_tensor = None
        sub_tensor_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(arg233_1, arg234_1)
        mul_tensor_2: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, arg235_1);  sub_tensor_1 = None
        unsqueeze_default_4: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1)
        unsqueeze_default_5: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_default_7: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None
        convert_element_type_default_1: "f16[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        add_tensor_2: "f16[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type_default, convert_element_type_default_1);  convert_element_type_default = convert_element_type_default_1 = None
        relu_default: "f16[128, 1408, 7, 7]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None
        le_scalar: "b8[128, 1408, 7, 7]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        full_default: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f16[128, 1408, 7, 7]" = torch.ops.aten.where.self(le_scalar, full_default, div_scalar);  le_scalar = full_default = div_scalar = None
        convert_element_type_default_2: "f32[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(where_self, torch.float32);  where_self = None
        squeeze_dims: "f32[1408]" = torch.ops.aten.squeeze.dims(arg234_1, [0, 2, 3]);  arg234_1 = None
        unsqueeze_default_8: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_9: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 2);  unsqueeze_default_8 = None
        unsqueeze_default_10: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 3);  unsqueeze_default_9 = None
        sum_dim_int_list: "f32[1408]" = torch.ops.aten.sum.dim_IntList(convert_element_type_default_2, [0, 2, 3])
        convert_element_type_default_3: "f32[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(arg233_1, torch.float32);  arg233_1 = None
        sub_tensor_2: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_default_3, unsqueeze_default_10);  convert_element_type_default_3 = unsqueeze_default_10 = None
        mul_tensor_4: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, sub_tensor_2)
        sum_dim_int_list_1: "f32[1408]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 2, 3]);  mul_tensor_4 = None
        mul_tensor_5: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00015943877551020407);  sum_dim_int_list = None
        unsqueeze_default_11: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_12: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 2);  unsqueeze_default_11 = None
        unsqueeze_default_13: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 3);  unsqueeze_default_12 = None
        mul_tensor_6: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00015943877551020407);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[1408]" = torch.ops.aten.squeeze.dims(arg235_1, [0, 2, 3]);  arg235_1 = None
        mul_tensor_7: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_8: "f32[1408]" = torch.ops.aten.mul.Tensor(mul_tensor_6, mul_tensor_7);  mul_tensor_6 = mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_tensor_8, 0);  mul_tensor_8 = None
        unsqueeze_default_15: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 2);  unsqueeze_default_14 = None
        unsqueeze_default_16: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 3);  unsqueeze_default_15 = None
        mul_tensor_9: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, arg61_1);  squeeze_dims_1 = arg61_1 = None
        unsqueeze_default_17: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_18: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 2);  unsqueeze_default_17 = None
        unsqueeze_default_19: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, 3);  unsqueeze_default_18 = None
        mul_tensor_10: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_16);  sub_tensor_2 = unsqueeze_default_16 = None
        sub_tensor_3: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, mul_tensor_10);  mul_tensor_10 = None
        sub_tensor_4: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_3, unsqueeze_default_13);  sub_tensor_3 = unsqueeze_default_13 = None
        mul_tensor_11: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_19);  sub_tensor_4 = unsqueeze_default_19 = None
        convert_element_type_default_4: "f16[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(mul_tensor_11, torch.float16);  mul_tensor_11 = None
        squeeze_dims_2: "f32[1408]" = torch.ops.aten.squeeze.dims(arg230_1, [0, 2, 3]);  arg230_1 = None
        unsqueeze_default_20: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(squeeze_dims_2, 0);  squeeze_dims_2 = None
        unsqueeze_default_21: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 2);  unsqueeze_default_20 = None
        unsqueeze_default_22: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_21, 3);  unsqueeze_default_21 = None
        sum_dim_int_list_2: "f32[1408]" = torch.ops.aten.sum.dim_IntList(convert_element_type_default_2, [0, 2, 3])
        convert_element_type_default_5: "f32[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(arg229_1, torch.float32);  arg229_1 = None
        sub_tensor_5: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_default_5, unsqueeze_default_22);  convert_element_type_default_5 = unsqueeze_default_22 = None
        mul_tensor_12: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, sub_tensor_5)
        sum_dim_int_list_3: "f32[1408]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 2, 3]);  mul_tensor_12 = None
        mul_tensor_13: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 0.00015943877551020407);  sum_dim_int_list_2 = None
        unsqueeze_default_23: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_24: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 2);  unsqueeze_default_23 = None
        unsqueeze_default_25: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, 3);  unsqueeze_default_24 = None
        mul_tensor_14: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 0.00015943877551020407);  sum_dim_int_list_3 = None
        squeeze_dims_3: "f32[1408]" = torch.ops.aten.squeeze.dims(arg231_1, [0, 2, 3]);  arg231_1 = None
        mul_tensor_15: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, squeeze_dims_3)
        mul_tensor_16: "f32[1408]" = torch.ops.aten.mul.Tensor(mul_tensor_14, mul_tensor_15);  mul_tensor_14 = mul_tensor_15 = None
        unsqueeze_default_26: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_tensor_16, 0);  mul_tensor_16 = None
        unsqueeze_default_27: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 2);  unsqueeze_default_26 = None
        unsqueeze_default_28: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_27, 3);  unsqueeze_default_27 = None
        mul_tensor_17: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, arg59_1);  squeeze_dims_3 = arg59_1 = None
        unsqueeze_default_29: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_tensor_17, 0);  mul_tensor_17 = None
        unsqueeze_default_30: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 2);  unsqueeze_default_29 = None
        unsqueeze_default_31: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, 3);  unsqueeze_default_30 = None
        mul_tensor_18: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_28);  sub_tensor_5 = unsqueeze_default_28 = None
        sub_tensor_6: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, mul_tensor_18);  convert_element_type_default_2 = mul_tensor_18 = None
        sub_tensor_7: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_6, unsqueeze_default_25);  sub_tensor_6 = unsqueeze_default_25 = None
        mul_tensor_19: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_7, unsqueeze_default_31);  sub_tensor_7 = unsqueeze_default_31 = None
        convert_element_type_default_6: "f16[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(mul_tensor_19, torch.float16);  mul_tensor_19 = None
        return (convert_element_type_default_4, convert_element_type_default_6)


def _default_make_inputs():
    return [
    torch.randn([128, 1408], dtype=torch.float16, device='cuda'),
    torch.randn(8830976, dtype=torch.float16, device='cuda').as_strided([128, 1408, 7, 7], [68992, 1, 9856, 1408]),  # arg229_1
    torch.randn([1, 1408, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1408, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn(8830976, dtype=torch.float16, device='cuda').as_strided([128, 1408, 7, 7], [68992, 1, 9856, 1408]),  # arg233_1
    torch.randn([1, 1408, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1408, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
