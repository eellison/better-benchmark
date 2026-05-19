"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf_cuda_h100-9-9-linux.aws.h100_graph75
Pattern hash: 6e2eef562da2
Shape hash: 8d501d44
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
    def forward(self, convolution: "f16[4, 64, 111, 111]", arg4_1: "f32[16]", arg3_1: "f32[16, 64, 1, 1]"):
        # No stacktrace found for following nodes
        relu_default: "f16[4, 64, 111, 111]" = torch.ops.aten.relu.default(convolution);  convolution = None
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_default, [3, 3], [2, 2], [0, 0], [1, 1], True);  relu_default = None
        getitem: "f16[4, 64, 55, 55]" = _low_memory_max_pool_with_offsets_default[0]
        getitem_1: "i8[4, 64, 55, 55]" = _low_memory_max_pool_with_offsets_default[1];  _low_memory_max_pool_with_offsets_default = None
        convert_element_type_default: "f16[16]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float16);  arg4_1 = None
        convert_element_type_default_1: "f16[16, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float16);  arg3_1 = None
        return (getitem, convert_element_type_default, convert_element_type_default_1, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([4, 64, 111, 111], dtype=torch.float16, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 64, 1, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
