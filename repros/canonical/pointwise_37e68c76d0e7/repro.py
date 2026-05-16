"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s1_g130
Pattern hash: 37e68c76d0e7
Shape hash: 5664c410
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 64]", arg1_1: "f32[10000, 64]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f16[64, 64]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float16);  arg0_1 = None
        convert_element_type_default_1: "f16[10000, 64]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float16);  arg1_1 = None
        permute_default: "f16[64, 64]" = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        return (convert_element_type_default_1, permute_default)


def _default_make_inputs():
    return [
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([10000, 64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
