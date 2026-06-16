"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_infer
Pattern hash: b71f0d447f2e
Shape hash: 25fb017b
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
    def forward(self, arg0_1: "bf16[192]", arg1_1: "bf16[128, 192, 17, 17]", arg2_1: "bf16[192]", arg3_1: "bf16[192]", arg4_1: "bf16[192]", arg5_1: "bf16[192]", arg6_1: "bf16[128, 192, 17, 17]", arg7_1: "bf16[192]", arg8_1: "bf16[192]", arg9_1: "bf16[192]", arg10_1: "bf16[192]", arg11_1: "bf16[128, 192, 17, 17]", arg12_1: "bf16[192]", arg13_1: "bf16[192]", arg14_1: "bf16[192]", arg15_1: "bf16[192]", arg16_1: "bf16[128, 192, 17, 17]", arg17_1: "bf16[192]", arg18_1: "bf16[192]", arg19_1: "bf16[192]"):
        # No stacktrace found for following nodes
        convert_element_type: "f32[192]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        unsqueeze: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
        unsqueeze_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        sub: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(arg1_1, unsqueeze_1);  arg1_1 = unsqueeze_1 = None
        convert_element_type_1: "f32[192]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        add: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_1, 0.001);  convert_element_type_1 = None
        sqrt: "f32[192]" = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
        unsqueeze_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_1: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_4: "bf16[192, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_5: "bf16[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_2: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        unsqueeze_6: "bf16[192, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_7: "bf16[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
        convert_element_type_2: "bf16[128, 192, 17, 17]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        relu: "bf16[128, 192, 17, 17]" = torch.ops.aten.relu.default(convert_element_type_2);  convert_element_type_2 = None
        convert_element_type_3: "f32[192]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        unsqueeze_8: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, -1);  convert_element_type_3 = None
        unsqueeze_9: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        sub_1: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(arg6_1, unsqueeze_9);  arg6_1 = unsqueeze_9 = None
        convert_element_type_4: "f32[192]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.float32);  arg7_1 = None
        add_2: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_4, 0.001);  convert_element_type_4 = None
        sqrt_1: "f32[192]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
        reciprocal_1: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        mul_3: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        unsqueeze_10: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
        unsqueeze_11: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        mul_4: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
        unsqueeze_12: "bf16[192, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
        unsqueeze_13: "bf16[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_5: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
        unsqueeze_14: "bf16[192, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_15: "bf16[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_3: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
        convert_element_type_5: "bf16[128, 192, 17, 17]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        relu_1: "bf16[128, 192, 17, 17]" = torch.ops.aten.relu.default(convert_element_type_5);  convert_element_type_5 = None
        convert_element_type_6: "f32[192]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.float32);  arg10_1 = None
        unsqueeze_16: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
        unsqueeze_17: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        sub_2: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(arg11_1, unsqueeze_17);  arg11_1 = unsqueeze_17 = None
        convert_element_type_7: "f32[192]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.float32);  arg12_1 = None
        add_4: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_7, 0.001);  convert_element_type_7 = None
        sqrt_2: "f32[192]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
        reciprocal_2: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        mul_6: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        unsqueeze_18: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
        unsqueeze_19: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        mul_7: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
        unsqueeze_20: "bf16[192, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
        unsqueeze_21: "bf16[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_8: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
        unsqueeze_22: "bf16[192, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_23: "bf16[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_5: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
        convert_element_type_8: "bf16[128, 192, 17, 17]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None
        relu_2: "bf16[128, 192, 17, 17]" = torch.ops.aten.relu.default(convert_element_type_8);  convert_element_type_8 = None
        convert_element_type_9: "f32[192]" = torch.ops.prims.convert_element_type.default(arg15_1, torch.float32);  arg15_1 = None
        unsqueeze_24: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_9, -1);  convert_element_type_9 = None
        unsqueeze_25: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        sub_3: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(arg16_1, unsqueeze_25);  arg16_1 = unsqueeze_25 = None
        convert_element_type_10: "f32[192]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.float32);  arg17_1 = None
        add_6: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_10, 0.001);  convert_element_type_10 = None
        sqrt_3: "f32[192]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
        reciprocal_3: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        mul_9: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        unsqueeze_26: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
        unsqueeze_27: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        mul_10: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
        unsqueeze_28: "bf16[192, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
        unsqueeze_29: "bf16[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_11: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
        unsqueeze_30: "bf16[192, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_31: "bf16[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_7: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
        convert_element_type_11: "bf16[128, 192, 17, 17]" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16);  add_7 = None
        relu_3: "bf16[128, 192, 17, 17]" = torch.ops.aten.relu.default(convert_element_type_11);  convert_element_type_11 = None
        cat: "bf16[128, 768, 17, 17]" = torch.ops.aten.cat.default([relu, relu_1, relu_2, relu_3], 1);  relu = relu_1 = relu_2 = relu_3 = None
        avg_pool2d: "bf16[128, 768, 17, 17]" = torch.ops.aten.avg_pool2d.default(cat, [3, 3], [1, 1], [1, 1])
        return (cat, avg_pool2d)



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
