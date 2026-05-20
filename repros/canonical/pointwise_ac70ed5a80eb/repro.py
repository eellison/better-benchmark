"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf-6-6-linux.aws.a100_graph31
Pattern hash: ac70ed5a80eb
Shape hash: 7d13c380
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([256, 1], f16), T([256, 1], f16))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f16[256, 1]", arg0_1: "f16[256, 1]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[256, 1]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        convert_element_type_default_1: "f32[256, 1]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        mul_tensor: "f32[256, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_default_1);  convert_element_type_default_1 = None
        sub_tensor: "f32[256, 1]" = torch.ops.aten.sub.Tensor(1, mul_tensor);  mul_tensor = None
        mul_tensor_1: "f32[256, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_default, sub_tensor);  convert_element_type_default = sub_tensor = None
        convert_element_type_default_2: "f16[256, 1]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float16);  mul_tensor_1 = None
        return convert_element_type_default_2


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
