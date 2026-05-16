"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g35
Pattern hash: da0425f904bd
Shape hash: 84b09002
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_64: "f16[32, 1024, 64]", bmm_66: "f16[32, 64, 1024]", bmm_67: "f16[32, 1024, 64]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_64, [4, 8, 1024, 64]);  bmm_64 = None
        reshape_default_1: "f16[4, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_66, [4, 8, 64, 1024]);  bmm_66 = None
        reshape_default_2: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_67, [4, 8, 1024, 64]);  bmm_67 = None
        permute_default: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None
        permute_default_1: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_3: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default, [4, 1024, 512]);  clone_default = None
        reshape_default_4: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_3, [4096, 512]);  reshape_default_3 = None
        permute_default_2: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_5: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_2, [4, 1024, 512]);  permute_default_2 = None
        clone_default_1: "f16[4, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_5, memory_format = torch.contiguous_format);  reshape_default_5 = None
        reshape_default_6: "f16[4096, 512]" = torch.ops.aten.reshape.default(clone_default_1, [4096, 512]);  clone_default_1 = None
        permute_default_3: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        clone_default_2: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        reshape_default_7: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_2, [4, 1024, 512]);  clone_default_2 = None
        reshape_default_8: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_7, [4096, 512]);  reshape_default_7 = None
        return (reshape_default_4, reshape_default_6, reshape_default_8)


def _default_make_inputs():
    return [
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
