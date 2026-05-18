"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s1_g5
Pattern hash: 9e288ad6925e
Shape hash: 3ac1fbc6
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
    def forward(self, addmm_66: "bf16[1024, 1024]", addmm_67: "bf16[1024, 1024]", addmm_68: "bf16[1024, 1024]", expand: "b8[1, 1, 1024, 1024]"):
        # No stacktrace found for following nodes
        reshape_default: "bf16[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_66, [1, 1024, 1024]);  addmm_66 = None
        reshape_default_1: "bf16[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(reshape_default, [1, 1024, -1, 64]);  reshape_default = None
        permute_default: "bf16[1, 16, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        reshape_default_2: "bf16[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_67, [1, 1024, 1024]);  addmm_67 = None
        reshape_default_3: "bf16[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_68, [1, 1024, 1024]);  addmm_68 = None
        reshape_default_4: "bf16[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(reshape_default_2, [1, 1024, -1, 64]);  reshape_default_2 = None
        permute_default_1: "bf16[1, 16, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None
        reshape_default_5: "bf16[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(reshape_default_3, [1, 1024, -1, 64]);  reshape_default_3 = None
        permute_default_2: "bf16[1, 16, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None
        full_default: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "bf16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_1, full_default);  expand = full_default_1 = full_default = None
        expand_default: "bf16[1, 16, 1024, 1024]" = torch.ops.aten.expand.default(where_self, [1, 16, 1024, 1024]);  where_self = None
        return (permute_default, permute_default_1, permute_default_2, expand_default)


def _default_make_inputs():
    return [
    torch.randn([1024, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randint(0, 2, [1, 1, 1024, 1024], dtype=torch.bool, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
