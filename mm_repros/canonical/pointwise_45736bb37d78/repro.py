"""
Standalone repro captured via capture_hook.
Label: nvidia_deeprecommender
Pattern hash: 45736bb37d78
Shape hash: c7536755
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
    def forward(self, arg0_1: "bf16[512, 197951]", arg1_1: "bf16[512]", arg2_1: "bf16[256, 197951]", arg3_1: "bf16[512, 512]", arg4_1: "bf16[512]", arg5_1: "bf16[1024, 512]", arg6_1: "bf16[1024]", arg7_1: "bf16[512, 1024]", arg8_1: "bf16[512]", arg9_1: "bf16[512, 512]", arg10_1: "bf16[512]", arg11_1: "bf16[197951, 512]", arg12_1: "bf16[197951]"):
        # No stacktrace found for following nodes
        permute: "bf16[197951, 512]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        addmm: "bf16[256, 512]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, permute);  arg1_1 = arg2_1 = permute = None
        convert_element_type: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32);  addmm = None
        gt: "b8[256, 512]" = torch.ops.aten.gt.Scalar(convert_element_type, 0)
        mul: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.0507009873554805)
        mul_1: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.0);  convert_element_type = None
        expm1: "f32[256, 512]" = torch.ops.aten.expm1.default(mul_1);  mul_1 = None
        mul_2: "f32[256, 512]" = torch.ops.aten.mul.Tensor(expm1, 1.7580993408473766);  expm1 = None
        where: "f32[256, 512]" = torch.ops.aten.where.self(gt, mul, mul_2);  gt = mul = mul_2 = None
        convert_element_type_1: "bf16[256, 512]" = torch.ops.prims.convert_element_type.default(where, torch.bfloat16);  where = None
        permute_1: "bf16[512, 512]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm_1: "bf16[256, 512]" = torch.ops.aten.addmm.default(arg4_1, convert_element_type_1, permute_1);  arg4_1 = convert_element_type_1 = permute_1 = None
        convert_element_type_2: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(addmm_1, torch.float32);  addmm_1 = None
        gt_1: "b8[256, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_2, 0)
        mul_3: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.0507009873554805)
        mul_4: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.0);  convert_element_type_2 = None
        expm1_1: "f32[256, 512]" = torch.ops.aten.expm1.default(mul_4);  mul_4 = None
        mul_5: "f32[256, 512]" = torch.ops.aten.mul.Tensor(expm1_1, 1.7580993408473766);  expm1_1 = None
        where_1: "f32[256, 512]" = torch.ops.aten.where.self(gt_1, mul_3, mul_5);  gt_1 = mul_3 = mul_5 = None
        convert_element_type_3: "bf16[256, 512]" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None
        permute_2: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_2: "bf16[256, 1024]" = torch.ops.aten.addmm.default(arg6_1, convert_element_type_3, permute_2);  arg6_1 = convert_element_type_3 = permute_2 = None
        convert_element_type_4: "f32[256, 1024]" = torch.ops.prims.convert_element_type.default(addmm_2, torch.float32);  addmm_2 = None
        gt_2: "b8[256, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_4, 0)
        mul_6: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.0507009873554805)
        mul_7: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.0);  convert_element_type_4 = None
        expm1_2: "f32[256, 1024]" = torch.ops.aten.expm1.default(mul_7);  mul_7 = None
        mul_8: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(expm1_2, 1.7580993408473766);  expm1_2 = None
        where_2: "f32[256, 1024]" = torch.ops.aten.where.self(gt_2, mul_6, mul_8);  gt_2 = mul_6 = mul_8 = None
        convert_element_type_5: "bf16[256, 1024]" = torch.ops.prims.convert_element_type.default(where_2, torch.bfloat16);  where_2 = None
        permute_3: "bf16[1024, 512]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_3: "bf16[256, 512]" = torch.ops.aten.addmm.default(arg8_1, convert_element_type_5, permute_3);  arg8_1 = convert_element_type_5 = permute_3 = None
        convert_element_type_6: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(addmm_3, torch.float32);  addmm_3 = None
        gt_3: "b8[256, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_6, 0)
        mul_9: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.0507009873554805)
        mul_10: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.0);  convert_element_type_6 = None
        expm1_3: "f32[256, 512]" = torch.ops.aten.expm1.default(mul_10);  mul_10 = None
        mul_11: "f32[256, 512]" = torch.ops.aten.mul.Tensor(expm1_3, 1.7580993408473766);  expm1_3 = None
        where_3: "f32[256, 512]" = torch.ops.aten.where.self(gt_3, mul_9, mul_11);  gt_3 = mul_9 = mul_11 = None
        convert_element_type_7: "bf16[256, 512]" = torch.ops.prims.convert_element_type.default(where_3, torch.bfloat16);  where_3 = None
        permute_4: "bf16[512, 512]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_4: "bf16[256, 512]" = torch.ops.aten.addmm.default(arg10_1, convert_element_type_7, permute_4);  arg10_1 = convert_element_type_7 = permute_4 = None
        convert_element_type_8: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(addmm_4, torch.float32);  addmm_4 = None
        gt_4: "b8[256, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_8, 0)
        mul_12: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.0507009873554805)
        mul_13: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.0);  convert_element_type_8 = None
        expm1_4: "f32[256, 512]" = torch.ops.aten.expm1.default(mul_13);  mul_13 = None
        mul_14: "f32[256, 512]" = torch.ops.aten.mul.Tensor(expm1_4, 1.7580993408473766);  expm1_4 = None
        where_4: "f32[256, 512]" = torch.ops.aten.where.self(gt_4, mul_12, mul_14);  gt_4 = mul_12 = mul_14 = None
        convert_element_type_9: "bf16[256, 512]" = torch.ops.prims.convert_element_type.default(where_4, torch.bfloat16);  where_4 = None
        permute_5: "bf16[512, 197951]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_5: "bf16[256, 197951]" = torch.ops.aten.addmm.default(arg12_1, convert_element_type_9, permute_5);  arg12_1 = convert_element_type_9 = permute_5 = None
        convert_element_type_10: "f32[256, 197951]" = torch.ops.prims.convert_element_type.default(addmm_5, torch.float32);  addmm_5 = None
        gt_5: "b8[256, 197951]" = torch.ops.aten.gt.Scalar(convert_element_type_10, 0)
        mul_15: "f32[256, 197951]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.0507009873554805)
        mul_16: "f32[256, 197951]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.0);  convert_element_type_10 = None
        expm1_5: "f32[256, 197951]" = torch.ops.aten.expm1.default(mul_16);  mul_16 = None
        mul_17: "f32[256, 197951]" = torch.ops.aten.mul.Tensor(expm1_5, 1.7580993408473766);  expm1_5 = None
        where_5: "f32[256, 197951]" = torch.ops.aten.where.self(gt_5, mul_15, mul_17);  gt_5 = mul_15 = mul_17 = None
        convert_element_type_11: "bf16[256, 197951]" = torch.ops.prims.convert_element_type.default(where_5, torch.bfloat16);  where_5 = None
        return convert_element_type_11



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
