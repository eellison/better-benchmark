"""
Standalone repro captured via capture_hook.
Label: nvidia_deeprecommender
Pattern hash: 4f2b279ce082
Shape hash: 01904a9e
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
    def forward(self, arg0_1: "f32[512]", arg1_1: "f32[512, 197951]", arg2_1: "f32[256, 197951]", arg3_1: "f32[512]", arg4_1: "f32[512, 512]", arg5_1: "f32[1024]", arg6_1: "f32[1024, 512]", arg7_1: "f32[512]", arg8_1: "f32[512, 1024]", arg9_1: "f32[512]", arg10_1: "f32[512, 512]", arg11_1: "f32[197951]", arg12_1: "f32[197951, 512]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[512, 197951]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_2: "bf16[256, 197951]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        permute: "bf16[197951, 512]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[256, 512]" = torch.ops.aten.addmm.default(convert_element_type, convert_element_type_2, permute);  convert_element_type = permute = None
        convert_element_type_3: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32)
        gt: "b8[256, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_3, 0)
        mul: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.0507009873554805)
        mul_1: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.0);  convert_element_type_3 = None
        expm1: "f32[256, 512]" = torch.ops.aten.expm1.default(mul_1);  mul_1 = None
        mul_2: "f32[256, 512]" = torch.ops.aten.mul.Tensor(expm1, 1.7580993408473766);  expm1 = None
        where: "f32[256, 512]" = torch.ops.aten.where.self(gt, mul, mul_2);  gt = mul = mul_2 = None
        convert_element_type_4: "bf16[256, 512]" = torch.ops.prims.convert_element_type.default(where, torch.bfloat16);  where = None
        convert_element_type_5: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_6: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        permute_1: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_6, [1, 0]);  convert_element_type_6 = None
        addmm_1: "bf16[256, 512]" = torch.ops.aten.addmm.default(convert_element_type_5, convert_element_type_4, permute_1);  convert_element_type_5 = None
        convert_element_type_7: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(addmm_1, torch.float32)
        gt_1: "b8[256, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_7, 0)
        mul_3: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.0507009873554805)
        mul_4: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.0);  convert_element_type_7 = None
        expm1_1: "f32[256, 512]" = torch.ops.aten.expm1.default(mul_4);  mul_4 = None
        mul_5: "f32[256, 512]" = torch.ops.aten.mul.Tensor(expm1_1, 1.7580993408473766);  expm1_1 = None
        where_1: "f32[256, 512]" = torch.ops.aten.where.self(gt_1, mul_3, mul_5);  gt_1 = mul_3 = mul_5 = None
        convert_element_type_8: "bf16[256, 512]" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None
        convert_element_type_9: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_10: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        permute_2: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_10, [1, 0]);  convert_element_type_10 = None
        addmm_2: "bf16[256, 1024]" = torch.ops.aten.addmm.default(convert_element_type_9, convert_element_type_8, permute_2);  convert_element_type_9 = None
        convert_element_type_11: "f32[256, 1024]" = torch.ops.prims.convert_element_type.default(addmm_2, torch.float32)
        gt_2: "b8[256, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_11, 0)
        mul_6: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.0507009873554805)
        mul_7: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.0);  convert_element_type_11 = None
        expm1_2: "f32[256, 1024]" = torch.ops.aten.expm1.default(mul_7);  mul_7 = None
        mul_8: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(expm1_2, 1.7580993408473766);  expm1_2 = None
        where_2: "f32[256, 1024]" = torch.ops.aten.where.self(gt_2, mul_6, mul_8);  gt_2 = mul_6 = mul_8 = None
        convert_element_type_12: "bf16[256, 1024]" = torch.ops.prims.convert_element_type.default(where_2, torch.bfloat16);  where_2 = None
        inductor_seeds: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[256, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_0, inductor_lookup_seed, 'rand');  _shape_param_0 = inductor_lookup_seed = None
        convert_element_type_13: "bf16[256, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt_3: "b8[256, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_13, 0.8);  convert_element_type_13 = None
        mul_9: "bf16[256, 1024]" = torch.ops.aten.mul.Tensor(gt_3, convert_element_type_12);  convert_element_type_12 = None
        mul_10: "bf16[256, 1024]" = torch.ops.aten.mul.Tensor(mul_9, 5.000000000000001);  mul_9 = None
        convert_element_type_14: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        convert_element_type_15: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        permute_3: "bf16[1024, 512]" = torch.ops.aten.permute.default(convert_element_type_15, [1, 0]);  convert_element_type_15 = None
        addmm_3: "bf16[256, 512]" = torch.ops.aten.addmm.default(convert_element_type_14, mul_10, permute_3);  convert_element_type_14 = None
        convert_element_type_16: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(addmm_3, torch.float32)
        gt_4: "b8[256, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_16, 0)
        mul_11: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.0507009873554805)
        mul_12: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.0);  convert_element_type_16 = None
        expm1_3: "f32[256, 512]" = torch.ops.aten.expm1.default(mul_12);  mul_12 = None
        mul_13: "f32[256, 512]" = torch.ops.aten.mul.Tensor(expm1_3, 1.7580993408473766);  expm1_3 = None
        where_3: "f32[256, 512]" = torch.ops.aten.where.self(gt_4, mul_11, mul_13);  gt_4 = mul_11 = mul_13 = None
        convert_element_type_17: "bf16[256, 512]" = torch.ops.prims.convert_element_type.default(where_3, torch.bfloat16);  where_3 = None
        convert_element_type_18: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        convert_element_type_19: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        permute_4: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_19, [1, 0]);  convert_element_type_19 = None
        addmm_4: "bf16[256, 512]" = torch.ops.aten.addmm.default(convert_element_type_18, convert_element_type_17, permute_4);  convert_element_type_18 = None
        convert_element_type_20: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(addmm_4, torch.float32)
        gt_5: "b8[256, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_20, 0)
        mul_14: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.0507009873554805)
        mul_15: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.0);  convert_element_type_20 = None
        expm1_4: "f32[256, 512]" = torch.ops.aten.expm1.default(mul_15);  mul_15 = None
        mul_16: "f32[256, 512]" = torch.ops.aten.mul.Tensor(expm1_4, 1.7580993408473766);  expm1_4 = None
        where_4: "f32[256, 512]" = torch.ops.aten.where.self(gt_5, mul_14, mul_16);  gt_5 = mul_14 = mul_16 = None
        convert_element_type_21: "bf16[256, 512]" = torch.ops.prims.convert_element_type.default(where_4, torch.bfloat16);  where_4 = None
        convert_element_type_22: "bf16[197951]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.bfloat16);  arg11_1 = None
        convert_element_type_23: "bf16[197951, 512]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.bfloat16);  arg12_1 = None
        permute_5: "bf16[512, 197951]" = torch.ops.aten.permute.default(convert_element_type_23, [1, 0]);  convert_element_type_23 = None
        addmm_5: "bf16[256, 197951]" = torch.ops.aten.addmm.default(convert_element_type_22, convert_element_type_21, permute_5);  convert_element_type_22 = None
        convert_element_type_24: "f32[256, 197951]" = torch.ops.prims.convert_element_type.default(addmm_5, torch.float32)
        gt_6: "b8[256, 197951]" = torch.ops.aten.gt.Scalar(convert_element_type_24, 0)
        mul_17: "f32[256, 197951]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.0507009873554805)
        mul_18: "f32[256, 197951]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.0);  convert_element_type_24 = None
        expm1_5: "f32[256, 197951]" = torch.ops.aten.expm1.default(mul_18);  mul_18 = None
        mul_19: "f32[256, 197951]" = torch.ops.aten.mul.Tensor(expm1_5, 1.7580993408473766);  expm1_5 = None
        where_5: "f32[256, 197951]" = torch.ops.aten.where.self(gt_6, mul_17, mul_19);  gt_6 = mul_17 = mul_19 = None
        convert_element_type_25: "bf16[256, 197951]" = torch.ops.prims.convert_element_type.default(where_5, torch.bfloat16);  where_5 = None
        permute_6: "bf16[197951, 512]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        permute_7: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        permute_8: "bf16[512, 1024]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_9: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_10: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (convert_element_type_2, addmm, convert_element_type_4, addmm_1, convert_element_type_8, addmm_2, gt_3, mul_10, addmm_3, convert_element_type_17, addmm_4, convert_element_type_21, addmm_5, convert_element_type_25, permute_6, permute_7, permute_8, permute_9, permute_10)



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
