"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: 544fac7c5583
Shape hash: f961de61
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
    def forward(self, arg0_1: "bf16[128, 384, 28, 28]", arg1_1: "bf16[128, 384, 28, 28]"):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 384, 28, 28]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type_1: "f32[128, 384, 28, 28]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        mul: "f32[128, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 0.7071067811865476)
        erf: "f32[128, 384, 28, 28]" = torch.ops.aten.erf.default(mul);  mul = None
        add: "f32[128, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_1: "f32[128, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_2: "f32[128, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type_1)
        mul_3: "f32[128, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_2, -0.5);  mul_2 = None
        exp: "f32[128, 384, 28, 28]" = torch.ops.aten.exp.default(mul_3);  mul_3 = None
        mul_4: "f32[128, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_5: "f32[128, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_4);  convert_element_type_1 = mul_4 = None
        add_1: "f32[128, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_1, mul_5);  mul_1 = mul_5 = None
        mul_6: "f32[128, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convert_element_type, add_1);  convert_element_type = add_1 = None
        convert_element_type_2: "bf16[128, 384, 28, 28]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        return convert_element_type_2



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
