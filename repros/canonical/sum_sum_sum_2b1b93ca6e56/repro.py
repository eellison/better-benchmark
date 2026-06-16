"""
Standalone repro captured via capture_hook.
Label: torchbench_functorch_dp_cifar10_train
Pattern hash: 2b1b93ca6e56
Shape hash: 9705fec3
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
    def forward(self, arg0_1: "bf16[128, 128, 4, 4]", arg1_1: "f32[128, 128, 4, 4]", arg2_1: "bf16[128, 128, 4, 4]", arg3_1: "f32[128, 32, 1, 1]", arg4_1: "f32[128, 32, 1, 1]", arg5_1: "f32[128]", arg6_1: "f32[128]", arg7_1: "bf16[128, 128, 4, 4]", arg8_1: "f32[128, 32, 1, 1]", arg9_1: "f32[128, 32, 1, 1]", arg10_1: "f32[128]", arg11_1: "f32[128]", arg12_1: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 128, 4, 4]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        add: "f32[128, 128, 4, 4]" = torch.ops.aten.add.Tensor(arg1_1, convert_element_type);  arg1_1 = convert_element_type = None
        convert_element_type_1: "f32[128, 128, 4, 4]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        view: "f32[128, 32, 4, 16]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  _shape_param_0 = None
        sub: "f32[128, 32, 4, 16]" = torch.ops.aten.sub.Tensor(view, arg3_1)
        mul: "f32[128, 32, 4, 16]" = torch.ops.aten.mul.Tensor(sub, arg4_1);  sub = None
        view_1: "f32[128, 128, 4, 4]" = torch.ops.aten.view.default(mul, _shape_param_1);  mul = _shape_param_1 = None
        unsqueeze: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(arg5_1, 0)
        unsqueeze_1: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2)
        unsqueeze_2: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_1: "f32[128, 128, 4, 4]" = torch.ops.aten.mul.Tensor(view_1, unsqueeze_2);  view_1 = unsqueeze_2 = None
        unsqueeze_3: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(arg6_1, 0);  arg6_1 = None
        unsqueeze_4: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        add_1: "f32[128, 128, 4, 4]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        convert_element_type_2: "f32[128, 128, 4, 4]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.float32);  arg7_1 = None
        view_2: "f32[128, 32, 4, 16]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_2);  _shape_param_2 = None
        sub_1: "f32[128, 32, 4, 16]" = torch.ops.aten.sub.Tensor(view_2, arg8_1)
        mul_2: "f32[128, 32, 4, 16]" = torch.ops.aten.mul.Tensor(sub_1, arg9_1);  sub_1 = None
        view_3: "f32[128, 128, 4, 4]" = torch.ops.aten.view.default(mul_2, _shape_param_3);  mul_2 = _shape_param_3 = None
        unsqueeze_6: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(arg10_1, 0)
        unsqueeze_7: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2)
        unsqueeze_8: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        mul_3: "f32[128, 128, 4, 4]" = torch.ops.aten.mul.Tensor(view_3, unsqueeze_8);  view_3 = unsqueeze_8 = None
        unsqueeze_9: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(arg11_1, 0);  arg11_1 = None
        unsqueeze_10: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 2);  unsqueeze_9 = None
        unsqueeze_11: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 3);  unsqueeze_10 = None
        add_2: "f32[128, 128, 4, 4]" = torch.ops.aten.add.Tensor(mul_3, unsqueeze_11);  mul_3 = unsqueeze_11 = None
        add_3: "f32[128, 128, 4, 4]" = torch.ops.aten.add.Tensor(add_1, add_2);  add_1 = add_2 = None
        relu: "f32[128, 128, 4, 4]" = torch.ops.aten.relu.default(add_3);  add_3 = None
        le: "b8[128, 128, 4, 4]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where: "f32[128, 128, 4, 4]" = torch.ops.aten.where.self(le, arg12_1, add);  le = arg12_1 = add = None
        mul_4: "f32[128, 128, 4, 4]" = torch.ops.aten.mul.Tensor(where, convert_element_type_2);  convert_element_type_2 = None
        view_4: "f32[128, 128, 16]" = torch.ops.aten.view.default(mul_4, _shape_param_4);  mul_4 = _shape_param_4 = None
        sum_1: "f32[128, 128]" = torch.ops.aten.sum.dim_IntList(view_4, [2]);  view_4 = None
        view_5: "f32[128, 128, 16]" = torch.ops.aten.view.default(where, _shape_param_5);  _shape_param_5 = None
        sum_2: "f32[128, 128]" = torch.ops.aten.sum.dim_IntList(view_5, [2]);  view_5 = None
        mul_5: "f32[128, 128]" = torch.ops.aten.mul.Tensor(sum_1, unsqueeze_6)
        view_6: "f32[128, 32, 4]" = torch.ops.aten.view.default(mul_5, _shape_param_6);  mul_5 = _shape_param_6 = None
        sum_3: "f32[128, 32]" = torch.ops.aten.sum.dim_IntList(view_6, [2]);  view_6 = None
        mul_6: "f32[128, 128]" = torch.ops.aten.mul.Tensor(sum_2, unsqueeze_6);  unsqueeze_6 = None
        view_7: "f32[128, 32, 4]" = torch.ops.aten.view.default(mul_6, _shape_param_7);  mul_6 = _shape_param_7 = None
        sum_4: "f32[128, 32]" = torch.ops.aten.sum.dim_IntList(view_7, [2]);  view_7 = None
        squeeze: "f32[128, 32]" = torch.ops.aten.squeeze.dims(arg9_1, [2, 3]);  arg9_1 = None
        unsqueeze_12: "f32[128, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze, -1)
        view_8: "f32[1, 32, 4]" = torch.ops.aten.view.default(arg10_1, _shape_param_8);  arg10_1 = _shape_param_8 = None
        mul_7: "f32[128, 32, 4]" = torch.ops.aten.mul.Tensor(unsqueeze_12, view_8);  view_8 = None
        squeeze_1: "f32[128, 32]" = torch.ops.aten.squeeze.dims(arg8_1, [2, 3]);  arg8_1 = None
        mul_8: "f32[128, 32]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_1)
        sub_2: "f32[128, 32]" = torch.ops.aten.sub.Tensor(mul_8, sum_3);  mul_8 = sum_3 = None
        mul_9: "f32[128, 32]" = torch.ops.aten.mul.Tensor(sub_2, squeeze);  sub_2 = None
        mul_10: "f32[128, 32]" = torch.ops.aten.mul.Tensor(mul_9, squeeze);  mul_9 = None
        mul_11: "f32[128, 32]" = torch.ops.aten.mul.Tensor(mul_10, squeeze);  mul_10 = None
        mul_12: "f32[128, 32]" = torch.ops.aten.mul.Tensor(mul_11, 0.015625);  mul_11 = None
        neg: "f32[128, 32]" = torch.ops.aten.neg.default(mul_12)
        mul_13: "f32[128, 32]" = torch.ops.aten.mul.Tensor(neg, squeeze_1);  neg = None
        mul_14: "f32[128, 32]" = torch.ops.aten.mul.Tensor(sum_4, squeeze);  sum_4 = squeeze = None
        mul_15: "f32[128, 32]" = torch.ops.aten.mul.Tensor(mul_14, 0.015625);  mul_14 = None
        sub_3: "f32[128, 32]" = torch.ops.aten.sub.Tensor(mul_13, mul_15);  mul_13 = mul_15 = None
        unsqueeze_13: "f32[128, 32, 4, 1]" = torch.ops.aten.unsqueeze.default(mul_7, -1);  mul_7 = None
        unsqueeze_14: "f32[128, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
        unsqueeze_15: "f32[128, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        unsqueeze_16: "f32[128, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_3, -1);  sub_3 = None
        unsqueeze_17: "f32[128, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        view_9: "f32[128, 32, 4, 16]" = torch.ops.aten.view.default(where, _shape_param_9);  _shape_param_9 = None
        mul_16: "f32[128, 32, 4, 16]" = torch.ops.aten.mul.Tensor(view_9, unsqueeze_13);  unsqueeze_13 = None
        mul_17: "f32[128, 32, 4, 16]" = torch.ops.aten.mul.Tensor(view_2, unsqueeze_15);  view_2 = unsqueeze_15 = None
        add_4: "f32[128, 32, 4, 16]" = torch.ops.aten.add.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None
        add_5: "f32[128, 32, 4, 16]" = torch.ops.aten.add.Tensor(add_4, unsqueeze_17);  add_4 = unsqueeze_17 = None
        view_10: "f32[128, 128, 4, 4]" = torch.ops.aten.view.default(add_5, _shape_param_10);  add_5 = _shape_param_10 = None
        view_11: "f32[128, 32, 4]" = torch.ops.aten.view.default(sum_1, _shape_param_11);  sum_1 = _shape_param_11 = None
        view_12: "f32[128, 32, 4]" = torch.ops.aten.view.default(sum_2, _shape_param_12);  _shape_param_12 = None
        unsqueeze_18: "f32[128, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_1, -1);  squeeze_1 = None
        mul_18: "f32[128, 32, 4]" = torch.ops.aten.mul.Tensor(view_12, unsqueeze_18);  unsqueeze_18 = None
        sub_4: "f32[128, 32, 4]" = torch.ops.aten.sub.Tensor(view_11, mul_18);  view_11 = mul_18 = None
        mul_19: "f32[128, 32, 4]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_12);  sub_4 = unsqueeze_12 = None
        sum_5: "f32[32, 4]" = torch.ops.aten.sum.dim_IntList(mul_19, [0]);  mul_19 = None
        view_13: "f32[128]" = torch.ops.aten.view.default(sum_5, _shape_param_13);  sum_5 = _shape_param_13 = None
        sum_6: "f32[128]" = torch.ops.aten.sum.dim_IntList(sum_2, [0])
        convert_element_type_3: "bf16[128, 128, 4, 4]" = torch.ops.prims.convert_element_type.default(view_10, torch.bfloat16);  view_10 = None
        mul_20: "f32[128, 128, 4, 4]" = torch.ops.aten.mul.Tensor(where, convert_element_type_1);  where = convert_element_type_1 = None
        view_14: "f32[128, 128, 16]" = torch.ops.aten.view.default(mul_20, _shape_param_14);  mul_20 = _shape_param_14 = None
        sum_7: "f32[128, 128]" = torch.ops.aten.sum.dim_IntList(view_14, [2]);  view_14 = None
        mul_21: "f32[128, 128]" = torch.ops.aten.mul.Tensor(sum_7, unsqueeze)
        view_15: "f32[128, 32, 4]" = torch.ops.aten.view.default(mul_21, _shape_param_15);  mul_21 = _shape_param_15 = None
        sum_8: "f32[128, 32]" = torch.ops.aten.sum.dim_IntList(view_15, [2]);  view_15 = None
        mul_22: "f32[128, 128]" = torch.ops.aten.mul.Tensor(sum_2, unsqueeze);  unsqueeze = None
        view_16: "f32[128, 32, 4]" = torch.ops.aten.view.default(mul_22, _shape_param_16);  mul_22 = _shape_param_16 = None
        sum_9: "f32[128, 32]" = torch.ops.aten.sum.dim_IntList(view_16, [2]);  view_16 = None
        squeeze_2: "f32[128, 32]" = torch.ops.aten.squeeze.dims(arg4_1, [2, 3]);  arg4_1 = None
        unsqueeze_19: "f32[128, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_2, -1)
        view_17: "f32[1, 32, 4]" = torch.ops.aten.view.default(arg5_1, _shape_param_17);  arg5_1 = _shape_param_17 = None
        mul_23: "f32[128, 32, 4]" = torch.ops.aten.mul.Tensor(unsqueeze_19, view_17);  view_17 = None
        squeeze_3: "f32[128, 32]" = torch.ops.aten.squeeze.dims(arg3_1, [2, 3]);  arg3_1 = None
        mul_24: "f32[128, 32]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_3)
        sub_5: "f32[128, 32]" = torch.ops.aten.sub.Tensor(mul_24, sum_8);  mul_24 = sum_8 = None
        mul_25: "f32[128, 32]" = torch.ops.aten.mul.Tensor(sub_5, squeeze_2);  sub_5 = None
        mul_26: "f32[128, 32]" = torch.ops.aten.mul.Tensor(mul_25, squeeze_2);  mul_25 = None
        mul_27: "f32[128, 32]" = torch.ops.aten.mul.Tensor(mul_26, squeeze_2);  mul_26 = None
        mul_28: "f32[128, 32]" = torch.ops.aten.mul.Tensor(mul_27, 0.015625);  mul_27 = None
        neg_1: "f32[128, 32]" = torch.ops.aten.neg.default(mul_28)
        mul_29: "f32[128, 32]" = torch.ops.aten.mul.Tensor(neg_1, squeeze_3);  neg_1 = None
        mul_30: "f32[128, 32]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_2);  sum_9 = squeeze_2 = None
        mul_31: "f32[128, 32]" = torch.ops.aten.mul.Tensor(mul_30, 0.015625);  mul_30 = None
        sub_6: "f32[128, 32]" = torch.ops.aten.sub.Tensor(mul_29, mul_31);  mul_29 = mul_31 = None
        unsqueeze_20: "f32[128, 32, 4, 1]" = torch.ops.aten.unsqueeze.default(mul_23, -1);  mul_23 = None
        unsqueeze_21: "f32[128, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_28, -1);  mul_28 = None
        unsqueeze_22: "f32[128, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_21, -1);  unsqueeze_21 = None
        unsqueeze_23: "f32[128, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_6, -1);  sub_6 = None
        unsqueeze_24: "f32[128, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_23, -1);  unsqueeze_23 = None
        mul_32: "f32[128, 32, 4, 16]" = torch.ops.aten.mul.Tensor(view_9, unsqueeze_20);  view_9 = unsqueeze_20 = None
        mul_33: "f32[128, 32, 4, 16]" = torch.ops.aten.mul.Tensor(view, unsqueeze_22);  view = unsqueeze_22 = None
        add_6: "f32[128, 32, 4, 16]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        add_7: "f32[128, 32, 4, 16]" = torch.ops.aten.add.Tensor(add_6, unsqueeze_24);  add_6 = unsqueeze_24 = None
        view_18: "f32[128, 128, 4, 4]" = torch.ops.aten.view.default(add_7, _shape_param_18);  add_7 = _shape_param_18 = None
        view_19: "f32[128, 32, 4]" = torch.ops.aten.view.default(sum_7, _shape_param_19);  sum_7 = _shape_param_19 = None
        unsqueeze_25: "f32[128, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_3, -1);  squeeze_3 = None
        mul_34: "f32[128, 32, 4]" = torch.ops.aten.mul.Tensor(view_12, unsqueeze_25);  view_12 = unsqueeze_25 = None
        sub_7: "f32[128, 32, 4]" = torch.ops.aten.sub.Tensor(view_19, mul_34);  view_19 = mul_34 = None
        mul_35: "f32[128, 32, 4]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_19);  sub_7 = unsqueeze_19 = None
        sum_10: "f32[32, 4]" = torch.ops.aten.sum.dim_IntList(mul_35, [0]);  mul_35 = None
        view_20: "f32[128]" = torch.ops.aten.view.default(sum_10, _shape_param_20);  sum_10 = _shape_param_20 = None
        sum_11: "f32[128]" = torch.ops.aten.sum.dim_IntList(sum_2, [0]);  sum_2 = None
        convert_element_type_4: "bf16[128, 128, 4, 4]" = torch.ops.prims.convert_element_type.default(view_18, torch.bfloat16);  view_18 = None
        return (view_13, sum_6, convert_element_type_3, view_20, sum_11, convert_element_type_4)



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
