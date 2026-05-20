"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf_cuda_h100-9-9-linux.aws.h100_graph75
Pattern hash: 1fee6f1bc148
Shape hash: c500e36b
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4, 128, 27, 27], f16), T([4, 128, 27, 27], f16), T([48], f32), T([48, 256, 1, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_11: "f16[4, 128, 27, 27]", convolution_12: "f16[4, 128, 27, 27]", arg28_1: "f32[48]", arg27_1: "f32[48, 256, 1, 1]"):
        # No stacktrace found for following nodes
        relu_default: "f16[4, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
        relu_default_1: "f16[4, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
        cat_default: "f16[4, 256, 27, 27]" = torch.ops.aten.cat.default([relu_default, relu_default_1], 1);  relu_default = relu_default_1 = None
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_default, [3, 3], [2, 2], [0, 0], [1, 1], True);  cat_default = None
        getitem: "f16[4, 256, 13, 13]" = _low_memory_max_pool_with_offsets_default[0]
        getitem_1: "i8[4, 256, 13, 13]" = _low_memory_max_pool_with_offsets_default[1];  _low_memory_max_pool_with_offsets_default = None
        convert_element_type_default: "f16[48]" = torch.ops.prims.convert_element_type.default(arg28_1, torch.float16);  arg28_1 = None
        convert_element_type_default_1: "f16[48, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(arg27_1, torch.float16);  arg27_1 = None
        return (getitem, convert_element_type_default, convert_element_type_default_1, getitem_1)


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
