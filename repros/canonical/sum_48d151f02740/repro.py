"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_train
Pattern hash: 48d151f02740
Shape hash: c9b4dcad
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
    def forward(self, arg0_1: "bf16[4, 128, 23923]", arg1_1: "bf16[4, 128, 23923]", arg2_1: "bf16[4, 256, 23923]"):
        # No stacktrace found for following nodes
        add: "bf16[4, 128, 23923]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[4, 128, 23923]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        convert_element_type_1: "f32[4, 256, 23923]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        slice_1: "f32[4, 128, 23923]" = torch.ops.aten.slice.Tensor(convert_element_type_1, 1, 0, 128)
        slice_2: "f32[4, 128, 23923]" = torch.ops.aten.slice.Tensor(convert_element_type_1, 1, 128, 256);  convert_element_type_1 = None
        sigmoid: "f32[4, 128, 23923]" = torch.ops.aten.sigmoid.default(slice_2);  slice_2 = None
        sub: "f32[4, 128, 23923]" = torch.ops.aten.sub.Tensor(1.0, sigmoid)
        mul: "f32[4, 128, 23923]" = torch.ops.aten.mul.Tensor(sub, sigmoid);  sub = None
        mul_1: "f32[4, 128, 23923]" = torch.ops.aten.mul.Tensor(mul, slice_1);  mul = slice_1 = None
        mul_2: "f32[4, 128, 23923]" = torch.ops.aten.mul.Tensor(mul_1, convert_element_type);  mul_1 = None
        mul_3: "f32[4, 128, 23923]" = torch.ops.aten.mul.Tensor(sigmoid, convert_element_type);  sigmoid = convert_element_type = None
        cat: "f32[4, 256, 23923]" = torch.ops.aten.cat.default([mul_3, mul_2], 1);  mul_3 = mul_2 = None
        convert_element_type_2: "bf16[4, 256, 23923]" = torch.ops.prims.convert_element_type.default(cat, torch.bfloat16);  cat = None
        sum_1: "bf16[256]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0, 2])
        convert_element_type_3: "f32[256]" = torch.ops.prims.convert_element_type.default(sum_1, torch.float32);  sum_1 = None
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
