"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g20
Pattern hash: ca2ce4eebd07
Shape hash: b775838b
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
    def forward(self, convolution: "f16[32, 768, 7, 7]", arg2_1: "f32[768]", arg3_1: "f32[50, 768]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[32, 768, 49]" = torch.ops.aten.reshape.default(convolution, [32, 768, 49]);  convolution = None
        permute_default: "f16[32, 49, 768]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None
        unsqueeze_default: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        expand_default: "f32[32, 1, 768]" = torch.ops.aten.expand.default(unsqueeze_default, [32, -1, -1]);  unsqueeze_default = None
        cat_default: "f32[32, 50, 768]" = torch.ops.aten.cat.default([expand_default, permute_default], 1);  expand_default = permute_default = None
        add_tensor: "f32[32, 50, 768]" = torch.ops.aten.add.Tensor(cat_default, arg3_1);  cat_default = arg3_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True);  add_tensor = None
        getitem: "f32[32, 50, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 50, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([32, 768, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([50, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
