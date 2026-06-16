"""
Standalone repro captured via capture_hook.
Label: torchbench_densenet121_train
Pattern hash: 9e1e02d0dd86
Shape hash: 0b922608
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[4, 256, 56, 56]", arg1_1: "bf16[4, 224, 56, 56]", arg2_1: "bf16[4, 192, 56, 56]", arg3_1: "bf16[4, 160, 56, 56]", arg4_1: "bf16[4, 128, 56, 56]", arg5_1: "bf16[4, 96, 56, 56]", arg6_1: "bf16[4, 64, 56, 56]", arg7_1: "bf16[]", arg8_1: "bf16[4, 64, 56, 56]", arg9_1: "bf16[4, 64, 56, 56]", arg10_1: "f32[1, 64, 1, 1]", arg11_1: "f32[64]", arg12_1: "f32[64]", arg13_1: "i8[4, 64, 56, 56]", arg14_1: "bf16[4, 64, 112, 112]", arg15_1: "f32[1, 64, 1, 1]", arg16_1: "f32[1, 64, 1, 1]", arg17_1: "f32[64]", arg18_1: "f32[64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        slice_1: "bf16[4, 64, 56, 56]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, 64);  arg0_1 = None
        slice_2: "bf16[4, 64, 56, 56]" = torch.ops.aten.slice.Tensor(arg1_1, 1, 0, 64);  arg1_1 = None
        add: "bf16[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(slice_1, slice_2);  slice_1 = slice_2 = None
        slice_3: "bf16[4, 64, 56, 56]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, 64);  arg2_1 = None
        add_1: "bf16[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(add, slice_3);  add = slice_3 = None
        slice_4: "bf16[4, 64, 56, 56]" = torch.ops.aten.slice.Tensor(arg3_1, 1, 0, 64);  arg3_1 = None
        add_2: "bf16[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(add_1, slice_4);  add_1 = slice_4 = None
        slice_5: "bf16[4, 64, 56, 56]" = torch.ops.aten.slice.Tensor(arg4_1, 1, 0, 64);  arg4_1 = None
        add_3: "bf16[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(add_2, slice_5);  add_2 = slice_5 = None
        slice_6: "bf16[4, 64, 56, 56]" = torch.ops.aten.slice.Tensor(arg5_1, 1, 0, 64);  arg5_1 = None
        add_4: "bf16[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(add_3, slice_6);  add_3 = slice_6 = None
        le: "b8[4, 64, 56, 56]" = torch.ops.aten.le.Scalar(arg6_1, 0);  arg6_1 = None
        where: "bf16[4, 64, 56, 56]" = torch.ops.aten.where.self(le, arg7_1, arg8_1);  le = arg8_1 = None
        convert_element_type: "f32[4, 64, 56, 56]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        sum_1: "f32[64]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 2, 3])
        convert_element_type_1: "f32[4, 64, 56, 56]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.float32);  arg9_1 = None
        sub: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg10_1);  convert_element_type_1 = arg10_1 = None
        mul: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(convert_element_type, sub)
        sum_2: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul, [0, 2, 3]);  mul = None
        mul_1: "f32[64]" = torch.ops.aten.mul.Tensor(sum_1, 7.971938775510203e-05)
        unsqueeze: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1, 0);  mul_1 = None
        unsqueeze_1: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_2: "f32[64]" = torch.ops.aten.mul.Tensor(sum_2, 7.971938775510203e-05)
        mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(arg11_1, arg11_1)
        mul_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        unsqueeze_3: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_4, 0);  mul_4 = None
        unsqueeze_4: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        mul_5: "f32[64]" = torch.ops.aten.mul.Tensor(arg11_1, arg12_1);  arg12_1 = None
        unsqueeze_6: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_5, 0);  mul_5 = None
        unsqueeze_7: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        mul_6: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_5);  sub = unsqueeze_5 = None
        sub_1: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convert_element_type, mul_6);  convert_element_type = mul_6 = None
        sub_2: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_1, unsqueeze_2);  sub_1 = unsqueeze_2 = None
        mul_7: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_8);  sub_2 = unsqueeze_8 = None
        mul_8: "f32[64]" = torch.ops.aten.mul.Tensor(sum_2, arg11_1);  sum_2 = arg11_1 = None
        convert_element_type_2: "bf16[4, 64, 56, 56]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        add_5: "bf16[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(add_4, convert_element_type_2);  add_4 = convert_element_type_2 = None
        full: "f32[256, 12544]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        view: "bf16[256, 3136]" = torch.ops.aten.view.default(add_5, _shape_param_1);  add_5 = _shape_param_1 = None
        _low_memory_max_pool_offsets_to_indices: "i64[4, 64, 56, 56]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(arg13_1, _shape_param_2, _shape_param_3, _shape_param_4, [1, 1], [1, 1]);  arg13_1 = _shape_param_2 = _shape_param_3 = _shape_param_4 = None
        view_1: "i64[256, 3136]" = torch.ops.aten.view.default(_low_memory_max_pool_offsets_to_indices, _shape_param_5);  _low_memory_max_pool_offsets_to_indices = _shape_param_5 = None
        convert_element_type_3: "f32[256, 3136]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        scatter_add: "f32[256, 12544]" = torch.ops.aten.scatter_add.default(full, 1, view_1, convert_element_type_3);  full = view_1 = convert_element_type_3 = None
        view_2: "f32[4, 64, 112, 112]" = torch.ops.aten.view.default(scatter_add, _shape_param_6);  scatter_add = _shape_param_6 = None
        convert_element_type_4: "bf16[4, 64, 112, 112]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        sub_3: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(arg14_1, arg15_1)
        mul_9: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, arg16_1);  sub_3 = None
        unsqueeze_9: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1)
        unsqueeze_10: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, -1);  unsqueeze_9 = None
        mul_10: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_9, unsqueeze_10);  mul_9 = unsqueeze_10 = None
        unsqueeze_11: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
        unsqueeze_12: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, -1);  unsqueeze_11 = None
        add_6: "f32[4, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_10, unsqueeze_12);  mul_10 = unsqueeze_12 = None
        convert_element_type_5: "bf16[4, 64, 112, 112]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        relu: "bf16[4, 64, 112, 112]" = torch.ops.aten.relu.default(convert_element_type_5);  convert_element_type_5 = None
        le_1: "b8[4, 64, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_1: "bf16[4, 64, 112, 112]" = torch.ops.aten.where.self(le_1, arg7_1, convert_element_type_4);  le_1 = arg7_1 = convert_element_type_4 = None
        convert_element_type_6: "f32[4, 64, 112, 112]" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        squeeze: "f32[64]" = torch.ops.aten.squeeze.dims(arg15_1, [0, 2, 3]);  arg15_1 = None
        unsqueeze_13: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_14: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 2);  unsqueeze_13 = None
        unsqueeze_15: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 3);  unsqueeze_14 = None
        sum_3: "f32[64]" = torch.ops.aten.sum.dim_IntList(convert_element_type_6, [0, 2, 3])
        convert_element_type_7: "f32[4, 64, 112, 112]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.float32);  arg14_1 = None
        sub_4: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convert_element_type_7, unsqueeze_15);  convert_element_type_7 = unsqueeze_15 = None
        mul_11: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(convert_element_type_6, sub_4)
        sum_4: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_11, [0, 2, 3]);  mul_11 = None
        mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(sum_3, 1.992984693877551e-05)
        unsqueeze_16: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_12, 0);  mul_12 = None
        unsqueeze_17: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 2);  unsqueeze_16 = None
        unsqueeze_18: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 3);  unsqueeze_17 = None
        mul_13: "f32[64]" = torch.ops.aten.mul.Tensor(sum_4, 1.992984693877551e-05)
        squeeze_1: "f32[64]" = torch.ops.aten.squeeze.dims(arg16_1, [0, 2, 3]);  arg16_1 = None
        mul_14: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_15: "f32[64]" = torch.ops.aten.mul.Tensor(mul_13, mul_14);  mul_13 = mul_14 = None
        unsqueeze_19: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_15, 0);  mul_15 = None
        unsqueeze_20: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_19, 2);  unsqueeze_19 = None
        unsqueeze_21: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, 3);  unsqueeze_20 = None
        mul_16: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, arg17_1);  arg17_1 = None
        unsqueeze_22: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_16, 0);  mul_16 = None
        unsqueeze_23: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 2);  unsqueeze_22 = None
        unsqueeze_24: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_23, 3);  unsqueeze_23 = None
        mul_17: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_21);  sub_4 = unsqueeze_21 = None
        sub_5: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convert_element_type_6, mul_17);  convert_element_type_6 = mul_17 = None
        sub_6: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_5, unsqueeze_18);  sub_5 = unsqueeze_18 = None
        mul_18: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_24);  sub_6 = unsqueeze_24 = None
        mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_1);  sum_4 = squeeze_1 = None
        convert_element_type_8: "bf16[4, 64, 112, 112]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        return (sum_1, mul_8, sum_3, mul_19, convert_element_type_8)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
