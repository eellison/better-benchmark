"""
Standalone repro captured via capture_hook.
Label: torchbench_lennard_jones_infer
Pattern hash: 38f32925c62e
Shape hash: c67d6e69
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
    def forward(self, arg0_1: "bf16[128, 1]", arg1_1: "bf16[16, 1]", arg2_1: "bf16[16]"):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 1]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        permute: "bf16[1, 16]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        convert_element_type_1: "f32[1, 16]" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        mul: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        mul_1: "f32[128, 16]" = torch.ops.aten.mul.Tensor(mul, 1);  mul = None
        convert_element_type_2: "f32[16]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul_2: "f32[16]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
        add: "f32[128, 16]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        convert_element_type_3: "bf16[128, 16]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        tanh: "bf16[128, 16]" = torch.ops.aten.tanh.default(convert_element_type_3);  convert_element_type_3 = None
        return tanh



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
