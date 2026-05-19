"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf_cuda_h100-4-7-linux.aws.h100_graph53
Pattern hash: 6fc7456d1274
Shape hash: 8cad7b96
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution_89: "f16[8, 240, 1, 1]", arg490_1: "f32[960]", arg489_1: "f32[960, 240, 1, 1]"):
        # No stacktrace found for following nodes
        relu_default: "f16[8, 240, 1, 1]" = torch.ops.aten.relu.default(convolution_89);  convolution_89 = None
        convert_element_type_default: "f16[960]" = torch.ops.prims.convert_element_type.default(arg490_1, torch.float16);  arg490_1 = None
        convert_element_type_default_1: "f16[960, 240, 1, 1]" = torch.ops.prims.convert_element_type.default(arg489_1, torch.float16);  arg489_1 = None
        return (relu_default, convert_element_type_default, convert_element_type_default_1)


def _default_make_inputs():
    return [
    torch.randn([8, 240, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960, 240, 1, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
