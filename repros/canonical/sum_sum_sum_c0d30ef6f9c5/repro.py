"""
Standalone repro captured via capture_hook.
Label: timm_repvgg_a2_train
Pattern hash: c0d30ef6f9c5
Shape hash: ac965caa
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
    def forward(self, arg0_1: "bf16[128, 1408]", arg1_1: "bf16[128, 1408, 7, 7]", arg2_1: "f32[1, 1408, 1, 1]", arg3_1: "f32[1, 1408, 1, 1]", arg4_1: "f32[1408]", arg5_1: "f32[1408]", arg6_1: "bf16[128, 1408, 7, 7]", arg7_1: "f32[1, 1408, 1, 1]", arg8_1: "f32[1, 1408, 1, 1]", arg9_1: "f32[1408]", arg10_1: "f32[1408]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        view: "bf16[128, 1408, 1, 1]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        squeeze: "bf16[128, 1408, 1]" = torch.ops.aten.squeeze.dim(view, 3);  view = None
        squeeze_1: "bf16[128, 1408]" = torch.ops.aten.squeeze.dim(squeeze, 2);  squeeze = None
        full: "bf16[180224]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        as_strided_scatter: "bf16[180224]" = torch.ops.aten.as_strided_scatter.default(full, squeeze_1, _shape_param_2, _shape_param_3, 0);  full = squeeze_1 = _shape_param_2 = _shape_param_3 = None
        as_strided: "bf16[128, 1408, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter, _shape_param_4, _shape_param_5, 0);  as_strided_scatter = _shape_param_4 = _shape_param_5 = None
        expand: "bf16[128, 1408, 7, 7]" = torch.ops.aten.expand.default(as_strided, _shape_param_6);  as_strided = _shape_param_6 = None
        div: "bf16[128, 1408, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        sub: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(arg1_1, arg2_1)
        mul: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub, arg3_1);  sub = None
        unsqueeze: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1)
        unsqueeze_1: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_3: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type: "bf16[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        sub_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(arg6_1, arg7_1)
        mul_2: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_1, arg8_1);  sub_1 = None
        unsqueeze_4: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1)
        unsqueeze_5: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_3: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul_2, unsqueeze_5);  mul_2 = unsqueeze_5 = None
        unsqueeze_6: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_7: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_3, unsqueeze_7);  mul_3 = unsqueeze_7 = None
        convert_element_type_1: "bf16[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        add_2: "bf16[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        relu: "bf16[128, 1408, 7, 7]" = torch.ops.aten.relu.default(add_2);  add_2 = None
        le: "b8[128, 1408, 7, 7]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[128, 1408, 7, 7]" = torch.ops.aten.where.self(le, full_1, div);  le = div = None
        convert_element_type_2: "f32[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze_2: "f32[1408]" = torch.ops.aten.squeeze.dims(arg7_1, [0, 2, 3]);  arg7_1 = None
        unsqueeze_8: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(squeeze_2, 0);  squeeze_2 = None
        unsqueeze_9: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 2);  unsqueeze_8 = None
        unsqueeze_10: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 3);  unsqueeze_9 = None
        sum_1: "f32[1408]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0, 2, 3])
        convert_element_type_3: "f32[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float32);  arg6_1 = None
        sub_2: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_3, unsqueeze_10);  convert_element_type_3 = unsqueeze_10 = None
        mul_4: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sub_2)
        sum_2: "f32[1408]" = torch.ops.aten.sum.dim_IntList(mul_4, [0, 2, 3]);  mul_4 = None
        mul_5: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_1, 0.00015943877551020407)
        unsqueeze_11: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_5, 0);  mul_5 = None
        unsqueeze_12: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 2);  unsqueeze_11 = None
        unsqueeze_13: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 3);  unsqueeze_12 = None
        mul_6: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_2, 0.00015943877551020407)
        squeeze_3: "f32[1408]" = torch.ops.aten.squeeze.dims(arg8_1, [0, 2, 3]);  arg8_1 = None
        mul_7: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_3, squeeze_3)
        mul_8: "f32[1408]" = torch.ops.aten.mul.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        unsqueeze_14: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_8, 0);  mul_8 = None
        unsqueeze_15: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 2);  unsqueeze_14 = None
        unsqueeze_16: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 3);  unsqueeze_15 = None
        mul_9: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_3, arg9_1);  arg9_1 = None
        unsqueeze_17: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_9, 0);  mul_9 = None
        unsqueeze_18: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 2);  unsqueeze_17 = None
        unsqueeze_19: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 3);  unsqueeze_18 = None
        mul_10: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_16);  sub_2 = unsqueeze_16 = None
        sub_3: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_2, mul_10);  mul_10 = None
        sub_4: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(sub_3, unsqueeze_13);  sub_3 = unsqueeze_13 = None
        mul_11: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_19);  sub_4 = unsqueeze_19 = None
        mul_12: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_2, squeeze_3);  sum_2 = squeeze_3 = None
        convert_element_type_4: "bf16[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None
        squeeze_4: "f32[1408]" = torch.ops.aten.squeeze.dims(arg2_1, [0, 2, 3]);  arg2_1 = None
        unsqueeze_20: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(squeeze_4, 0);  squeeze_4 = None
        unsqueeze_21: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, 2);  unsqueeze_20 = None
        unsqueeze_22: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_21, 3);  unsqueeze_21 = None
        sum_3: "f32[1408]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0, 2, 3])
        convert_element_type_5: "f32[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        sub_5: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_5, unsqueeze_22);  convert_element_type_5 = unsqueeze_22 = None
        mul_13: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sub_5)
        sum_4: "f32[1408]" = torch.ops.aten.sum.dim_IntList(mul_13, [0, 2, 3]);  mul_13 = None
        mul_14: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_3, 0.00015943877551020407)
        unsqueeze_23: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_14, 0);  mul_14 = None
        unsqueeze_24: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_23, 2);  unsqueeze_23 = None
        unsqueeze_25: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 3);  unsqueeze_24 = None
        mul_15: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_4, 0.00015943877551020407)
        squeeze_5: "f32[1408]" = torch.ops.aten.squeeze.dims(arg3_1, [0, 2, 3]);  arg3_1 = None
        mul_16: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_5, squeeze_5)
        mul_17: "f32[1408]" = torch.ops.aten.mul.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        unsqueeze_26: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_17, 0);  mul_17 = None
        unsqueeze_27: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 2);  unsqueeze_26 = None
        unsqueeze_28: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_27, 3);  unsqueeze_27 = None
        mul_18: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_5, arg4_1);  arg4_1 = None
        unsqueeze_29: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_18, 0);  mul_18 = None
        unsqueeze_30: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_29, 2);  unsqueeze_29 = None
        unsqueeze_31: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 3);  unsqueeze_30 = None
        mul_19: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_28);  sub_5 = unsqueeze_28 = None
        sub_6: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_2, mul_19);  convert_element_type_2 = mul_19 = None
        sub_7: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(sub_6, unsqueeze_25);  sub_6 = unsqueeze_25 = None
        mul_20: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_31);  sub_7 = unsqueeze_31 = None
        mul_21: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_5);  sum_4 = squeeze_5 = None
        convert_element_type_6: "bf16[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None
        return (full_1, sum_1, mul_12, convert_element_type_4, sum_3, mul_21, convert_element_type_6)



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
