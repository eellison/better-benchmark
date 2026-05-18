"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g36
Pattern hash: c92e63431b58
Shape hash: e4f0eff6
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
    def forward(self, bmm_40: "f16[12, 512, 64]", bmm_42: "f16[12, 64, 512]", bmm_43: "f16[12, 512, 64]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_40, [1, 12, 512, 64]);  bmm_40 = None
        reshape_default_1: "f16[1, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_42, [1, 12, 64, 512]);  bmm_42 = None
        reshape_default_2: "f16[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_43, [1, 12, 512, 64]);  bmm_43 = None
        permute_default: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None
        permute_default_1: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_3: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_default, [1, 512, 768]);  clone_default = None
        reshape_default_4: "f16[512, 768]" = torch.ops.aten.reshape.default(reshape_default_3, [512, 768]);  reshape_default_3 = None
        permute_default_2: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_5: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(permute_default_2, [1, 512, 768]);  permute_default_2 = None
        reshape_default_6: "f16[512, 768]" = torch.ops.aten.reshape.default(reshape_default_5, [512, 768]);  reshape_default_5 = None
        permute_default_3: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        clone_default_1: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        reshape_default_7: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_default_1, [1, 512, 768]);  clone_default_1 = None
        reshape_default_8: "f16[512, 768]" = torch.ops.aten.reshape.default(reshape_default_7, [512, 768]);  reshape_default_7 = None
        return (reshape_default_4, reshape_default_6, reshape_default_8)


def _default_make_inputs():
    return [
    torch.randn([12, 512, 64], dtype=torch.float16, device='cuda'),
    torch.randn([12, 64, 512], dtype=torch.float16, device='cuda'),
    torch.randn([12, 512, 64], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
