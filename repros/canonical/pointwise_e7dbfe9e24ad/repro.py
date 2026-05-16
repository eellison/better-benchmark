"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g21
Pattern hash: e7dbfe9e24ad
Shape hash: 849da001
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_23: "f16[12, 512, 64]", arg194_1: "f32[768]", arg193_1: "f32[768, 768]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_23, [1, 12, 512, 64]);  bmm_23 = None
        permute_default: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_default, [1, 512, -1]);  clone_default = None
        convert_element_type_default: "f16[768]" = torch.ops.prims.convert_element_type.default(arg194_1, torch.float16);  arg194_1 = None
        convert_element_type_default_1: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(arg193_1, torch.float16);  arg193_1 = None
        reshape_default_2: "f16[512, 768]" = torch.ops.aten.reshape.default(reshape_default_1, [512, 768]);  reshape_default_1 = None
        permute_default_1: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        return (convert_element_type_default, reshape_default_2, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([12, 512, 64], dtype=torch.float16, device='cuda'),
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
