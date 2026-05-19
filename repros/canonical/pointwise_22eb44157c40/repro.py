"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf-4-6-linux.aws.a100_graph24
Pattern hash: 22eb44157c40
Shape hash: 7f35ccee
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
    def forward(self, arg0_1: "f32[32, 3, 3, 3]", arg1_1: "f32[32, 3, 224, 224]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f16[32, 3, 3, 3]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float16);  arg0_1 = None
        convert_element_type_default_1: "f16[32, 3, 224, 224]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float16);  arg1_1 = None
        return (convert_element_type_default, convert_element_type_default_1)


def _default_make_inputs():
    return [
    torch.randn([32, 3, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([32, 3, 224, 224], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
