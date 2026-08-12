"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train
Pattern hash: 38fa7c1b328d
Shape hash: c4d61901
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
    def forward(self, arg0_1: "bf16[4096, 512]", arg1_1: "f32[32, 128, 512]", arg2_1: "bf16[4096, 512]", arg3_1: "bf16[4096, 512]", arg4_1: "bf16[4096, 512]", arg5_1: "bf16[4096, 512]", arg6_1: "bf16[4096, 512]", arg7_1: "bf16[4096, 512]", arg8_1: "bf16[4096, 512]", arg9_1: "bf16[4096, 512]", arg10_1: "bf16[4096, 512]", arg11_1: "bf16[4096, 512]", arg12_1: "bf16[4096, 512]", arg13_1: "bf16[4096, 512]", arg14_1: "bf16[4096, 512]", arg15_1: "bf16[4096, 512]", arg16_1: "bf16[4096, 512]", arg17_1: "b8[32, 128, 512]", arg18_1: "f32[512]", arg19_1: "f32[32, 128, 512]", arg20_1: "f32[32, 128, 1]", arg21_1: "b8[32, 128, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18):
        # No stacktrace found for following nodes
        view: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        add: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(arg1_1, convert_element_type);  arg1_1 = convert_element_type = None
        view_1: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        convert_element_type_1: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add_1: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add, convert_element_type_1);  add = convert_element_type_1 = None
        view_2: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        convert_element_type_2: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add_2: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_1, convert_element_type_2);  add_1 = convert_element_type_2 = None
        view_3: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg4_1, _shape_param_3);  arg4_1 = _shape_param_3 = None
        convert_element_type_3: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        add_3: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_2, convert_element_type_3);  add_2 = convert_element_type_3 = None
        view_4: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg5_1, _shape_param_4);  arg5_1 = _shape_param_4 = None
        convert_element_type_4: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        add_4: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_3, convert_element_type_4);  add_3 = convert_element_type_4 = None
        view_5: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg6_1, _shape_param_5);  arg6_1 = _shape_param_5 = None
        convert_element_type_5: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        add_5: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_4, convert_element_type_5);  add_4 = convert_element_type_5 = None
        view_6: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg7_1, _shape_param_6);  arg7_1 = _shape_param_6 = None
        convert_element_type_6: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        add_6: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_5, convert_element_type_6);  add_5 = convert_element_type_6 = None
        view_7: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg8_1, _shape_param_7);  arg8_1 = _shape_param_7 = None
        convert_element_type_7: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        add_7: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_6, convert_element_type_7);  add_6 = convert_element_type_7 = None
        view_8: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg9_1, _shape_param_8);  arg9_1 = _shape_param_8 = None
        convert_element_type_8: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        add_8: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_7, convert_element_type_8);  add_7 = convert_element_type_8 = None
        view_9: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg10_1, _shape_param_9);  arg10_1 = _shape_param_9 = None
        convert_element_type_9: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_9, torch.float32);  view_9 = None
        add_9: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_8, convert_element_type_9);  add_8 = convert_element_type_9 = None
        view_10: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg11_1, _shape_param_10);  arg11_1 = _shape_param_10 = None
        convert_element_type_10: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        add_10: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_9, convert_element_type_10);  add_9 = convert_element_type_10 = None
        view_11: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg12_1, _shape_param_11);  arg12_1 = _shape_param_11 = None
        convert_element_type_11: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        add_11: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_10, convert_element_type_11);  add_10 = convert_element_type_11 = None
        view_12: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg13_1, _shape_param_12);  arg13_1 = _shape_param_12 = None
        convert_element_type_12: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_12, torch.float32);  view_12 = None
        add_12: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_11, convert_element_type_12);  add_11 = convert_element_type_12 = None
        view_13: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg14_1, _shape_param_13);  arg14_1 = _shape_param_13 = None
        convert_element_type_13: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        add_13: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_12, convert_element_type_13);  add_12 = convert_element_type_13 = None
        view_14: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg15_1, _shape_param_14);  arg15_1 = _shape_param_14 = None
        convert_element_type_14: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_14, torch.float32);  view_14 = None
        add_14: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_13, convert_element_type_14);  add_13 = convert_element_type_14 = None
        view_15: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg16_1, _shape_param_15);  arg16_1 = _shape_param_15 = None
        convert_element_type_15: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_15, torch.float32);  view_15 = None
        add_15: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_14, convert_element_type_15);  add_14 = convert_element_type_15 = None
        convert_element_type_16: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.float32);  arg17_1 = None
        mul: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
        mul_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_15, mul);  add_15 = mul = None
        mul_2: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_1, arg18_1);  arg18_1 = None
        mul_3: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg19_1, arg20_1)
        mul_4: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_1, mul_3);  mul_1 = mul_3 = None
        sum_1: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_4, [0, 1], True, dtype = torch.float32);  mul_4 = None
        view_16: "f32[512]" = torch.ops.aten.view.default(sum_1, _shape_param_16);  sum_1 = _shape_param_16 = None
        mul_5: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_2, arg19_1)
        mul_6: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_2, arg20_1);  mul_2 = None
        sum_2: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [2], True, dtype = torch.float32);  mul_5 = None
        pow_1: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg20_1, 3);  arg20_1 = None
        mul_7: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_2, -0.5);  sum_2 = None
        mul_8: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_7, pow_1);  mul_7 = pow_1 = None
        expand: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_8, _shape_param_17);  mul_8 = _shape_param_17 = None
        div: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand, 512);  expand = None
        pow_2: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg19_1, 1.0);  arg19_1 = None
        mul_9: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_2, 2.0);  pow_2 = None
        mul_10: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div, mul_9);  div = mul_9 = None
        add_16: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(mul_6, mul_10);  mul_6 = mul_10 = None
        convert_element_type_17: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16)
        convert_element_type_18: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg21_1, torch.bfloat16);  arg21_1 = None
        mul_11: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
        mul_12: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_17, mul_11);  convert_element_type_17 = mul_11 = None
        view_17: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_12, _shape_param_18);  mul_12 = _shape_param_18 = None
        permute: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_17, [1, 0])
        return (view_16, add_16, view_17, permute)



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
