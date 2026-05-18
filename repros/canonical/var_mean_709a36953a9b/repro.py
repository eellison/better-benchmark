"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s2_g20
Pattern hash: 709a36953a9b
Shape hash: ae0ee0ba
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
    def forward(self, convolution: "f16[8, 768, 14, 14]", arg4_1: "f32[1, 1, 768]", arg5_1: "f32[1, 1, 768]", arg3_1: "f32[1, 198, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        reshape_default: "f16[8, 768, 196]" = torch.ops.aten.reshape.default(convolution, _shape_param_0);  convolution = _shape_param_0 = None
        permute_default: "f16[8, 196, 768]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None
        expand_default: "f32[8, 1, 768]" = torch.ops.aten.expand.default(arg4_1, _shape_param_1);  arg4_1 = _shape_param_1 = None
        expand_default_1: "f32[8, 1, 768]" = torch.ops.aten.expand.default(arg5_1, _shape_param_2);  arg5_1 = _shape_param_2 = None
        cat_default: "f32[8, 198, 768]" = torch.ops.aten.cat.default([expand_default, expand_default_1, permute_default], 1);  expand_default = expand_default_1 = permute_default = None
        add_tensor: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(cat_default, arg3_1);  cat_default = arg3_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True);  add_tensor = None
        getitem: "f32[8, 198, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 198, 1]" = var_mean_correction[1];  var_mean_correction = None
        _output_to_half_0: "f16[8, 198, 1]" = torch.ops.prims.convert_element_type.default(getitem, torch.float16);  getitem = None
        _output_to_half_1: "f16[8, 198, 1]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float16);  getitem_1 = None
        return (_output_to_half_0, _output_to_half_1)


def _default_make_inputs():
    return [
    torch.randn([8, 768, 14, 14], dtype=torch.float16, device='cuda'),
    torch.randn([1, 1, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 198, 768], dtype=torch.float32, device='cuda'),
    [8, 768, 196],  # _shape_param_0
    [8, -1, -1],  # _shape_param_1
    [8, -1, -1],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
