"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s3_g20
Pattern hash: 1dd6a8ce4db1
Shape hash: 50a741e2
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
    def forward(self, convolution: "f16[128, 192, 14, 14]", arg5_1: "f32[1, 1, 192]", arg4_1: "f32[1, 197, 192]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[128, 192, 196]" = torch.ops.aten.reshape.default(convolution, [128, 192, 196]);  convolution = None
        permute_default: "f16[128, 196, 192]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None
        expand_default: "f32[128, 1, 192]" = torch.ops.aten.expand.default(arg5_1, [128, -1, -1]);  arg5_1 = None
        cat_default: "f32[128, 197, 192]" = torch.ops.aten.cat.default([expand_default, permute_default], 1);  expand_default = permute_default = None
        add_tensor: "f32[128, 197, 192]" = torch.ops.aten.add.Tensor(cat_default, arg4_1);  cat_default = arg4_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True);  add_tensor = None
        getitem: "f32[128, 197, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 197, 1]" = var_mean_correction[1];  var_mean_correction = None
        _output_to_half_0: "f16[128, 197, 1]" = torch.ops.prims.convert_element_type.default(getitem, torch.float16);  getitem = None
        _output_to_half_1: "f16[128, 197, 1]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float16);  getitem_1 = None
        return (_output_to_half_0, _output_to_half_1)


def _default_make_inputs():
    return [
    torch.randn([128, 192, 14, 14], dtype=torch.float16, device='cuda'),
    torch.randn([1, 1, 192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 197, 192], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
