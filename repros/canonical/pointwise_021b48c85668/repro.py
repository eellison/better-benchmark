"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g21
Pattern hash: 021b48c85668
Shape hash: 32710e6a
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
    def forward(self, arg12_1: "f32[768]", arg11_1: "f32[768, 768]", getitem_2: "f32[1, 512, 768]", arg14_1: "f32[768]", arg13_1: "f32[768, 768]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f16[768]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.float16);  arg12_1 = None
        convert_element_type_default_1: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.float16);  arg11_1 = None
        convert_element_type_default_2: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_2, torch.float16);  getitem_2 = None
        reshape_default: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_2, [512, 768]);  convert_element_type_default_2 = None
        permute_default: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        convert_element_type_default_3: "f16[768]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.float16);  arg14_1 = None
        convert_element_type_default_4: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.float16);  arg13_1 = None
        permute_default_1: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        return (convert_element_type_default, reshape_default, permute_default, convert_element_type_default_3, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
