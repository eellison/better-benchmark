"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s3_g62
Pattern hash: da62f1349a33
Shape hash: 76e8bfec
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
    def forward(self, addmm_66: "f16[1904, 768]", addmm_67: "f16[1904, 768]", arg189_1: "f32[768]", arg188_1: "f32[768, 768]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 476, 768]" = torch.ops.aten.reshape.default(addmm_66, [4, 476, 768]);  addmm_66 = None
        reshape_default_1: "f16[4, 476, 768]" = torch.ops.aten.reshape.default(addmm_67, [4, 476, 768]);  addmm_67 = None
        convert_element_type_default: "f16[768]" = torch.ops.prims.convert_element_type.default(arg189_1, torch.float16);  arg189_1 = None
        convert_element_type_default_1: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(arg188_1, torch.float16);  arg188_1 = None
        permute_default: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        reshape_default_2: "f16[4, 476, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, [4, 476, 12, 64]);  reshape_default = None
        permute_default_1: "f16[4, 12, 476, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        reshape_default_3: "f16[4, 476, 12, 64]" = torch.ops.aten.reshape.default(reshape_default_1, [4, 476, 12, 64]);  reshape_default_1 = None
        permute_default_2: "f16[4, 12, 476, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None
        permute_default_3: "f16[4, 12, 64, 476]" = torch.ops.aten.permute.default(permute_default_2, [0, 1, 3, 2]);  permute_default_2 = None
        expand_default: "f16[4, 12, 476, 64]" = torch.ops.aten.expand.default(permute_default_1, [4, 12, 476, 64]);  permute_default_1 = None
        clone_default: "f16[4, 12, 476, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_4: "f16[48, 476, 64]" = torch.ops.aten.reshape.default(clone_default, [48, 476, 64]);  clone_default = None
        expand_default_1: "f16[4, 12, 64, 476]" = torch.ops.aten.expand.default(permute_default_3, [4, 12, 64, 476]);  permute_default_3 = None
        clone_default_1: "f16[4, 12, 64, 476]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_5: "f16[48, 64, 476]" = torch.ops.aten.reshape.default(clone_default_1, [48, 64, 476]);  clone_default_1 = None
        return (convert_element_type_default, permute_default, reshape_default_4, reshape_default_5)


def _default_make_inputs():
    return [
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
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
