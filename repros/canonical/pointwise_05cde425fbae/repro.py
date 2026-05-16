"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s2_g20
Pattern hash: 05cde425fbae
Shape hash: 0374652f
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "f32[768]", arg1_1: "f32[768, 3, 16, 16]", arg0_1: "f32[8, 3, 224, 224]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f16[768]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float16);  arg2_1 = None
        convert_element_type_default_1: "f16[768, 3, 16, 16]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float16);  arg1_1 = None
        convert_element_type_default_2: "f16[8, 3, 224, 224]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float16);  arg0_1 = None
        return (convert_element_type_default, convert_element_type_default_1, convert_element_type_default_2)


def _default_make_inputs():
    return [
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3, 16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([8, 3, 224, 224], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
