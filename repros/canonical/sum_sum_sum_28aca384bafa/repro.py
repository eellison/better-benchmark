"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train
Pattern hash: 28aca384bafa
Shape hash: bfd27ee1
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
    def forward(self, arg0_1: "bf16[128, 1536, 12, 12]", arg1_1: "bf16[128, 1536, 12, 12]", arg2_1: "bf16[128, 1536, 12, 12]", arg3_1: "bf16[128, 1536, 1, 1]", arg4_1: "bf16[128, 1536, 12, 12]", arg5_1: "f32[]"):
        # No stacktrace found for following nodes
        mul: "bf16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(arg0_1, 0.9622504486493761);  arg0_1 = None
        mul_1: "bf16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul, 1.7015043497085571);  mul = None
        convert_element_type: "f32[128, 1536, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_1, torch.float32);  mul_1 = None
        convert_element_type_1: "f32[128, 1536, 12, 12]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        mul_2: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 0.7071067811865476)
        erf: "f32[128, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_2);  mul_2 = None
        add: "f32[128, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_3: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_4: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type_1)
        mul_5: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_4, -0.5);  mul_4 = None
        exp: "f32[128, 1536, 12, 12]" = torch.ops.aten.exp.default(mul_5);  mul_5 = None
        mul_6: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_7: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_6);  convert_element_type_1 = mul_6 = None
        add_1: "f32[128, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_3, mul_7);  mul_3 = mul_7 = None
        mul_8: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type, add_1);  convert_element_type = add_1 = None
        convert_element_type_2: "bf16[128, 1536, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        add_2: "bf16[128, 1536, 12, 12]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type_2);  arg2_1 = convert_element_type_2 = None
        mul_9: "bf16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_2, 0.2)
        sigmoid: "bf16[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(arg3_1);  arg3_1 = None
        mul_10: "bf16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(arg4_1, sigmoid)
        mul_11: "bf16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_10, 2.0);  mul_10 = None
        mul_12: "bf16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_9, mul_11);  mul_11 = None
        mul_13: "bf16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_9, arg5_1);  mul_9 = arg5_1 = None
        sum_1: "f32[]" = torch.ops.aten.sum.default(mul_12, dtype = torch.float32);  mul_12 = None
        mul_14: "bf16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_13, 2.0);  mul_13 = None
        mul_15: "bf16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_14, arg4_1);  arg4_1 = None
        sum_2: "f32[128, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_15, [2, 3], True, dtype = torch.float32);  mul_15 = None
        convert_element_type_3: "bf16[128, 1536, 1, 1]" = torch.ops.prims.convert_element_type.default(sum_2, torch.bfloat16);  sum_2 = None
        convert_element_type_4: "f32[128, 1536, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        convert_element_type_5: "f32[128, 1536, 1, 1]" = torch.ops.prims.convert_element_type.default(sigmoid, torch.float32)
        sub: "f32[128, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, convert_element_type_5)
        mul_16: "f32[128, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_5, sub);  convert_element_type_5 = sub = None
        mul_17: "f32[128, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_4, mul_16);  convert_element_type_4 = mul_16 = None
        convert_element_type_6: "bf16[128, 1536, 1, 1]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        sum_3: "bf16[1536]" = torch.ops.aten.sum.dim_IntList(convert_element_type_6, [0, 2, 3])
        convert_element_type_7: "f32[1536]" = torch.ops.prims.convert_element_type.default(sum_3, torch.float32);  sum_3 = None
        return (add_2, sigmoid, sum_1, mul_14, convert_element_type_6, convert_element_type_7)



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
