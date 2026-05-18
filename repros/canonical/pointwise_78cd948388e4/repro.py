"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s2_g21
Pattern hash: 78cd948388e4
Shape hash: 316438ff
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
    def forward(self, getitem_126: "f16[4, 12, 1024, 64]", arg141_1: "f32[768, 768]", arg140_1: "f32[768]"):
        # No stacktrace found for following nodes
        permute_default: "f16[4, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
        reshape_default: "f16[4, 1024, 768]" = torch.ops.aten.reshape.default(permute_default, [4, 1024, -1]);  permute_default = None
        reshape_default_1: "f16[4096, 768]" = torch.ops.aten.reshape.default(reshape_default, [-1, 768]);  reshape_default = None
        convert_element_type_default: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(arg141_1, torch.float16);  arg141_1 = None
        convert_element_type_default_1: "f16[768]" = torch.ops.prims.convert_element_type.default(arg140_1, torch.float16);  arg140_1 = None
        return (reshape_default_1, convert_element_type_default, convert_element_type_default_1)


def _default_make_inputs():
    return [
    torch.randn(3145728, dtype=torch.float16, device='cuda').as_strided([4, 12, 1024, 64], [786432, 64, 768, 1]),  # getitem_126
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
