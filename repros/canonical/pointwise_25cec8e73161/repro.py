"""
Standalone repro captured via capture_hook.
Label: hf_openai/gpt-oss-20b_infer
Pattern hash: 25cec8e73161
Shape hash: 55c3c977
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
    def forward(self, arg0_1: "i64[4000]", arg1_1: "bf16[32, 5760]", arg2_1: "bf16[4000, 5760]"):
        # No stacktrace found for following nodes
        clamp_max: "i64[4000]" = torch.ops.aten.clamp_max.default(arg0_1, 31);  arg0_1 = None
        index: "bf16[4000, 5760]" = torch.ops.aten.index.Tensor(arg1_1, [clamp_max]);  arg1_1 = None
        add: "bf16[4000, 5760]" = torch.ops.aten.add.Tensor(arg2_1, index);  arg2_1 = index = None
        slice_1: "bf16[4000, 2880]" = torch.ops.aten.slice.Tensor(add, 1, 1, 9223372036854775807, 2)
        convert_element_type: "f32[4000, 2880]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        clamp_min: "f32[4000, 2880]" = torch.ops.aten.clamp_min.default(convert_element_type, -7.0);  convert_element_type = None
        clamp_max_1: "f32[4000, 2880]" = torch.ops.aten.clamp_max.default(clamp_min, 7.0);  clamp_min = None
        convert_element_type_1: "bf16[4000, 2880]" = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.bfloat16);  clamp_max_1 = None
        add_1: "bf16[4000, 2880]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1);  convert_element_type_1 = None
        slice_2: "bf16[4000, 2880]" = torch.ops.aten.slice.Tensor(add, 1, 0, 9223372036854775807, 2);  add = None
        convert_element_type_2: "f32[4000, 2880]" = torch.ops.prims.convert_element_type.default(slice_2, torch.float32);  slice_2 = None
        clamp_max_2: "f32[4000, 2880]" = torch.ops.aten.clamp_max.default(convert_element_type_2, 7.0);  convert_element_type_2 = None
        convert_element_type_3: "bf16[4000, 2880]" = torch.ops.prims.convert_element_type.default(clamp_max_2, torch.bfloat16);  clamp_max_2 = None
        mul: "bf16[4000, 2880]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.702)
        sigmoid: "bf16[4000, 2880]" = torch.ops.aten.sigmoid.default(mul);  mul = None
        mul_1: "bf16[4000, 2880]" = torch.ops.aten.mul.Tensor(convert_element_type_3, sigmoid);  convert_element_type_3 = sigmoid = None
        mul_2: "bf16[4000, 2880]" = torch.ops.aten.mul.Tensor(add_1, mul_1);  add_1 = mul_1 = None
        return (clamp_max, mul_2)



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
