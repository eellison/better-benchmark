"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_infer
Pattern hash: fb4ca2518d23
Shape hash: 53dca1d5
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
    def forward(self, arg0_1: "bf16[128, 1536, 1, 1]", arg1_1: "bf16[128, 1536, 9, 9]", arg2_1: "bf16[128, 1536, 9, 9]"):
        # No stacktrace found for following nodes
        sigmoid: "bf16[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(arg0_1);  arg0_1 = None
        mul: "bf16[128, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(arg1_1, sigmoid);  arg1_1 = sigmoid = None
        mul_1: "bf16[128, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul, 2.0);  mul = None
        mul_2: "bf16[128, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_1, 0.2);  mul_1 = None
        add: "bf16[128, 1536, 9, 9]" = torch.ops.aten.add.Tensor(mul_2, arg2_1);  mul_2 = arg2_1 = None
        convert_element_type: "f32[128, 1536, 9, 9]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        neg: "f32[128, 1536, 9, 9]" = torch.ops.aten.neg.default(convert_element_type)
        exp: "f32[128, 1536, 9, 9]" = torch.ops.aten.exp.default(neg);  neg = None
        add_1: "f32[128, 1536, 9, 9]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[128, 1536, 9, 9]" = torch.ops.aten.div.Tensor(convert_element_type, add_1);  convert_element_type = add_1 = None
        convert_element_type_1: "bf16[128, 1536, 9, 9]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        mul_3: "bf16[128, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 0.9622504486493761);  convert_element_type_1 = None
        return (add, mul_3)



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
