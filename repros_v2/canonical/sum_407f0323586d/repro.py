"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train
Pattern hash: 407f0323586d
Shape hash: 97f7c01b
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
    def forward(self, arg0_1: "bf16[128, 128, 56, 56]", arg1_1: "bf16[128, 128, 56, 56]", arg2_1: "bf16[128, 128, 56, 56]"):
        # No stacktrace found for following nodes
        add: "bf16[128, 128, 56, 56]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        mul: "bf16[128, 128, 56, 56]" = torch.ops.aten.mul.Tensor(add, 1.0);  add = None
        convert_element_type: "f32[128, 128, 56, 56]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        convert_element_type_1: "f32[128, 128, 56, 56]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        sigmoid: "f32[128, 128, 56, 56]" = torch.ops.aten.sigmoid.default(convert_element_type_1)
        mul_1: "f32[128, 128, 56, 56]" = torch.ops.aten.mul.Tensor(convert_element_type, sigmoid);  convert_element_type = None
        sub: "f32[128, 128, 56, 56]" = torch.ops.aten.sub.Tensor(1, sigmoid);  sigmoid = None
        mul_2: "f32[128, 128, 56, 56]" = torch.ops.aten.mul.Tensor(convert_element_type_1, sub);  convert_element_type_1 = sub = None
        add_1: "f32[128, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_2, 1);  mul_2 = None
        mul_3: "f32[128, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_1, add_1);  mul_1 = add_1 = None
        convert_element_type_2: "bf16[128, 128, 56, 56]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
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
