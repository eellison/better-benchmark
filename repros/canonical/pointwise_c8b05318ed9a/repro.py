"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf_cuda_h100-9-9-linux.aws.h100_graph75
Pattern hash: c8b05318ed9a
Shape hash: 7f27dcde
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4, 64, 13, 13], f16), T([256], f32), T([256, 64, 1, 1], f32), T([256], f32), T([256, 64, 3, 3], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_22: "f16[4, 64, 13, 13]", arg48_1: "f32[256]", arg47_1: "f32[256, 64, 1, 1]", arg50_1: "f32[256]", arg49_1: "f32[256, 64, 3, 3]"):
        # No stacktrace found for following nodes
        relu_default: "f16[4, 64, 13, 13]" = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None
        convert_element_type_default: "f16[256]" = torch.ops.prims.convert_element_type.default(arg48_1, torch.float16);  arg48_1 = None
        convert_element_type_default_1: "f16[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(arg47_1, torch.float16);  arg47_1 = None
        convert_element_type_default_2: "f16[256]" = torch.ops.prims.convert_element_type.default(arg50_1, torch.float16);  arg50_1 = None
        convert_element_type_default_3: "f16[256, 64, 3, 3]" = torch.ops.prims.convert_element_type.default(arg49_1, torch.float16);  arg49_1 = None
        return (relu_default, convert_element_type_default, convert_element_type_default_1, convert_element_type_default_2, convert_element_type_default_3)


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
