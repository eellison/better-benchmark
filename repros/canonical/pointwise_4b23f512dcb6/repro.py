"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g20
Pattern hash: 4b23f512dcb6
Shape hash: afe8a2ac
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
    def forward(self, bmm_11: "f16[32, 1024, 64]", arg47_1: "f32[512, 512]", mm_36: "f16[4096, 512]", mm_37: "f16[4096, 512]", arg57_1: "f32[512, 512]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_11, [4, 8, 1024, 64]);  bmm_11 = None
        permute_default: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default, [4, 1024, -1]);  clone_default = None
        convert_element_type_default: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(arg47_1, torch.float16);  arg47_1 = None
        permute_default_1: "f16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        reshape_default_2: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_1, [4096, 512]);  reshape_default_1 = None
        reshape_default_3: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_36, [4, 1024, 512]);  mm_36 = None
        reshape_default_4: "f16[4, 1024, 8, 64]" = torch.ops.aten.reshape.default(reshape_default_3, [4, 1024, -1, 64]);  reshape_default_3 = None
        permute_default_2: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None
        reshape_default_5: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_37, [4, 1024, 512]);  mm_37 = None
        reshape_default_6: "f16[4, 1024, 8, 64]" = torch.ops.aten.reshape.default(reshape_default_5, [4, 1024, -1, 64]);  reshape_default_5 = None
        permute_default_3: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_6, [0, 2, 1, 3]);  reshape_default_6 = None
        convert_element_type_default_1: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(arg57_1, torch.float16);  arg57_1 = None
        permute_default_4: "f16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        permute_default_5: "f16[4, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_default_3, [0, 1, 3, 2]);  permute_default_3 = None
        expand_default: "f16[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_default_2, [4, 8, 1024, 64]);  permute_default_2 = None
        clone_default_1: "f16[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_7: "f16[32, 1024, 64]" = torch.ops.aten.reshape.default(clone_default_1, [32, 1024, 64]);  clone_default_1 = None
        expand_default_1: "f16[4, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_default_5, [4, 8, 64, 1024]);  permute_default_5 = None
        clone_default_2: "f16[4, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_8: "f16[32, 64, 1024]" = torch.ops.aten.reshape.default(clone_default_2, [32, 64, 1024]);  clone_default_2 = None
        return (permute_default_1, reshape_default_2, permute_default_4, reshape_default_7, reshape_default_8)


def _default_make_inputs():
    return [
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
