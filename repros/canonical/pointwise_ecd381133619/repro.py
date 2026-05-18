"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s2_g20
Pattern hash: ecd381133619
Shape hash: 98eebebb
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 3, 7, 7]", arg2_1: "f32[4, 3, 224, 224]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f16[64, 3, 7, 7]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float16);  arg0_1 = None
        convert_element_type_default_1: "f16[4, 3, 224, 224]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float16);  arg2_1 = None
        return (convert_element_type_default, convert_element_type_default_1)


def _default_make_inputs():
    return [
    torch.randn([64, 3, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([4, 3, 224, 224], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
