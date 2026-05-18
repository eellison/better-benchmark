"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g35
Pattern hash: 110b84367814
Shape hash: bfaca2cb
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
    def forward(self, mm_115: "f16[4096, 512]", mm_187: "f16[4096, 512]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_115, [4, 1024, 512]);  mm_115 = None
        reshape_default_1: "f16[4, 1024, 8, 64]" = torch.ops.aten.reshape.default(reshape_default, [4, 1024, 8, 64]);  reshape_default = None
        permute_default: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        clone_default: "f16[4, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_2: "f16[32, 1024, 64]" = torch.ops.aten.reshape.default(clone_default, [32, 1024, 64]);  clone_default = None
        reshape_default_3: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_187, [4, 1024, 512]);  mm_187 = None
        reshape_default_4: "f16[4, 1024, 8, 64]" = torch.ops.aten.reshape.default(reshape_default_3, [4, 1024, 8, 64]);  reshape_default_3 = None
        permute_default_1: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None
        clone_default_1: "f16[4, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_5: "f16[32, 1024, 64]" = torch.ops.aten.reshape.default(clone_default_1, [32, 1024, 64]);  clone_default_1 = None
        return (reshape_default_2, reshape_default_5)


def _default_make_inputs():
    return [
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
