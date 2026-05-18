"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s7_g20
Pattern hash: 50db0aab6f24
Shape hash: a50b64b8
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
    def forward(self, convolution_42: "f16[128, 1408, 7, 7]", convolution_43: "f16[128, 1408, 7, 7]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(convolution_42, torch.float32);  convolution_42 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[1, 1408, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 1408, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        convert_element_type_default_1: "f32[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(convolution_43, torch.float32);  convolution_43 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_1, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_default_1 = None
        getitem_2: "f32[1, 1408, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 1408, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        return (getitem, getitem_1, getitem_2, getitem_3)


def _default_make_inputs():
    return [
    torch.randn([128, 1408, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([128, 1408, 7, 7], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
