"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: f2eb6784cabb
Shape hash: ff30dd34
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
    def forward(self, arg0_1: "bf16[128, 192, 28, 28]", arg1_1: "bf16[128, 192, 28, 28]", arg2_1: "f32[1, 192, 1, 1]", arg3_1: "f32[1, 192, 1, 1]", arg4_1: "f32[192]", arg5_1: "f32[192]", arg6_1: "f32[1, 192, 28, 28]", arg7_1: "f32[1, 192, 1, 1]", arg8_1: "f32[192]", arg9_1: "f32[192]", arg10_1: "f32[128, 192, 28, 28]"):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 192, 28, 28]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        sum_1: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 2, 3])
        sub: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(arg1_1, arg2_1)
        mul: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub, arg3_1);  sub = None
        unsqueeze: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1)
        unsqueeze_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_1: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add: "f32[128, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type_1: "bf16[128, 192, 28, 28]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        add_1: "f32[128, 192, 28, 28]" = torch.ops.aten.add.Tensor(convert_element_type_1, arg6_1);  convert_element_type_1 = arg6_1 = None
        sub_1: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_1, arg7_1);  add_1 = arg7_1 = None
        mul_2: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(convert_element_type, sub_1)
        sum_2: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_2, [0, 2, 3]);  mul_2 = None
        mul_3: "f32[192]" = torch.ops.aten.mul.Tensor(sum_1, 9.964923469387754e-06)
        unsqueeze_4: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_3, 0);  mul_3 = None
        unsqueeze_5: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        unsqueeze_6: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None
        mul_4: "f32[192]" = torch.ops.aten.mul.Tensor(sum_2, 9.964923469387754e-06)
        mul_5: "f32[192]" = torch.ops.aten.mul.Tensor(arg8_1, arg8_1)
        mul_6: "f32[192]" = torch.ops.aten.mul.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze_7: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_6, 0);  mul_6 = None
        unsqueeze_8: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None
        unsqueeze_9: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 3);  unsqueeze_8 = None
        mul_7: "f32[192]" = torch.ops.aten.mul.Tensor(arg8_1, arg9_1);  arg9_1 = None
        unsqueeze_10: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_7, 0);  mul_7 = None
        unsqueeze_11: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        unsqueeze_12: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 3);  unsqueeze_11 = None
        mul_8: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_9);  sub_1 = unsqueeze_9 = None
        sub_2: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convert_element_type, mul_8);  convert_element_type = mul_8 = None
        sub_3: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_2, unsqueeze_6);  sub_2 = unsqueeze_6 = None
        mul_9: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_12);  sub_3 = unsqueeze_12 = None
        mul_10: "f32[192]" = torch.ops.aten.mul.Tensor(sum_2, arg8_1);  sum_2 = arg8_1 = None
        add_2: "f32[128, 192, 28, 28]" = torch.ops.aten.add.Tensor(arg10_1, mul_9);  arg10_1 = mul_9 = None
        convert_element_type_2: "bf16[128, 192, 28, 28]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16)
        sum_3: "f32[1, 192, 28, 28]" = torch.ops.aten.sum.dim_IntList(add_2, [0], True, dtype = torch.float32);  add_2 = None
        convert_element_type_3: "f32[128, 192, 28, 28]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        squeeze: "f32[192]" = torch.ops.aten.squeeze.dims(arg2_1, [0, 2, 3]);  arg2_1 = None
        unsqueeze_13: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_14: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 2);  unsqueeze_13 = None
        unsqueeze_15: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 3);  unsqueeze_14 = None
        sum_4: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_3, [0, 2, 3])
        convert_element_type_4: "f32[128, 192, 28, 28]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        sub_4: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convert_element_type_4, unsqueeze_15);  convert_element_type_4 = unsqueeze_15 = None
        mul_11: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(convert_element_type_3, sub_4)
        sum_5: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_11, [0, 2, 3]);  mul_11 = None
        mul_12: "f32[192]" = torch.ops.aten.mul.Tensor(sum_4, 9.964923469387754e-06)
        unsqueeze_16: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_12, 0);  mul_12 = None
        unsqueeze_17: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 2);  unsqueeze_16 = None
        unsqueeze_18: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 3);  unsqueeze_17 = None
        mul_13: "f32[192]" = torch.ops.aten.mul.Tensor(sum_5, 9.964923469387754e-06)
        squeeze_1: "f32[192]" = torch.ops.aten.squeeze.dims(arg3_1, [0, 2, 3]);  arg3_1 = None
        mul_14: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_15: "f32[192]" = torch.ops.aten.mul.Tensor(mul_13, mul_14);  mul_13 = mul_14 = None
        unsqueeze_19: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_15, 0);  mul_15 = None
        unsqueeze_20: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_19, 2);  unsqueeze_19 = None
        unsqueeze_21: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, 3);  unsqueeze_20 = None
        mul_16: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_1, arg4_1);  arg4_1 = None
        unsqueeze_22: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_16, 0);  mul_16 = None
        unsqueeze_23: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 2);  unsqueeze_22 = None
        unsqueeze_24: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_23, 3);  unsqueeze_23 = None
        mul_17: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_21);  sub_4 = unsqueeze_21 = None
        sub_5: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convert_element_type_3, mul_17);  convert_element_type_3 = mul_17 = None
        sub_6: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_5, unsqueeze_18);  sub_5 = unsqueeze_18 = None
        mul_18: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_24);  sub_6 = unsqueeze_24 = None
        mul_19: "f32[192]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_1);  sum_5 = squeeze_1 = None
        convert_element_type_5: "bf16[128, 192, 28, 28]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        sum_6: "bf16[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_5, [0, 2, 3])
        convert_element_type_6: "f32[192]" = torch.ops.prims.convert_element_type.default(sum_6, torch.float32);  sum_6 = None
        return (sum_1, mul_10, sum_3, sum_4, mul_19, convert_element_type_5, convert_element_type_6)



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
