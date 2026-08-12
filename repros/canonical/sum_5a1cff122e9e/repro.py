"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train
Pattern hash: 5a1cff122e9e
Shape hash: 8e74fca9
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
    def forward(self, arg0_1: "bf16[128, 128, 48, 48]", arg1_1: "bf16[128, 128, 48, 48]", arg2_1: "bf16[128, 128, 48, 48]"):
        # No stacktrace found for following nodes
        add: "bf16[128, 128, 48, 48]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        mul: "bf16[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(add, 1.0);  add = None
        mul_1: "bf16[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul, 1.7015043497085571);  mul = None
        convert_element_type: "f32[128, 128, 48, 48]" = torch.ops.prims.convert_element_type.default(mul_1, torch.float32);  mul_1 = None
        convert_element_type_1: "f32[128, 128, 48, 48]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul_2: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 0.7071067811865476)
        erf: "f32[128, 128, 48, 48]" = torch.ops.aten.erf.default(mul_2);  mul_2 = None
        add_1: "f32[128, 128, 48, 48]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_3: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(add_1, 0.5);  add_1 = None
        mul_4: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type_1)
        mul_5: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_4, -0.5);  mul_4 = None
        exp: "f32[128, 128, 48, 48]" = torch.ops.aten.exp.default(mul_5);  mul_5 = None
        mul_6: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_7: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_6);  convert_element_type_1 = mul_6 = None
        add_2: "f32[128, 128, 48, 48]" = torch.ops.aten.add.Tensor(mul_3, mul_7);  mul_3 = mul_7 = None
        mul_8: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convert_element_type, add_2);  convert_element_type = add_2 = None
        convert_element_type_2: "bf16[128, 128, 48, 48]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        sum_1: "bf16[128]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0, 2, 3])
        convert_element_type_3: "f32[128]" = torch.ops.prims.convert_element_type.default(sum_1, torch.float32);  sum_1 = None
        return (convert_element_type_2, convert_element_type_3)



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
