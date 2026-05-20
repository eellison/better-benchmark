"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf_cuda_h100-9-9-linux.aws.h100_graph75
Pattern hash: 185bb1780a34
Shape hash: e43b3a1a
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4, 256, 13, 13], f16), T([4, 256, 13, 13], f16), T([1000], f32), T([1000, 512, 1, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_23: "f16[4, 256, 13, 13]", convolution_24: "f16[4, 256, 13, 13]", arg52_1: "f32[1000]", arg51_1: "f32[1000, 512, 1, 1]"):
        # No stacktrace found for following nodes
        relu_default: "f16[4, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_23);  convolution_23 = None
        relu_default_1: "f16[4, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_24);  convolution_24 = None
        cat_default: "f16[4, 512, 13, 13]" = torch.ops.aten.cat.default([relu_default, relu_default_1], 1);  relu_default = relu_default_1 = None
        convert_element_type_default: "f16[1000]" = torch.ops.prims.convert_element_type.default(arg52_1, torch.float16);  arg52_1 = None
        convert_element_type_default_1: "f16[1000, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(arg51_1, torch.float16);  arg51_1 = None
        return (cat_default, convert_element_type_default, convert_element_type_default_1)


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
