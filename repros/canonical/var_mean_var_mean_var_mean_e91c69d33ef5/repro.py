"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s7_g20
Pattern hash: e91c69d33ef5
Shape hash: ac926dd4
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
    def forward(self, relu_19: "f16[128, 384, 14, 14]", convolution_40: "f16[128, 384, 14, 14]", convolution_41: "f16[128, 384, 14, 14]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[128, 384, 14, 14]" = torch.ops.prims.convert_element_type.default(relu_19, torch.float32);  relu_19 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[1, 384, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 384, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        convert_element_type_default_1: "f32[128, 384, 14, 14]" = torch.ops.prims.convert_element_type.default(convolution_40, torch.float32);  convolution_40 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_1, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_default_1 = None
        getitem_2: "f32[1, 384, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 384, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        convert_element_type_default_2: "f32[128, 384, 14, 14]" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32);  convolution_41 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_default_2 = None
        getitem_4: "f32[1, 384, 1, 1]" = var_mean_correction_2[0]
        getitem_5: "f32[1, 384, 1, 1]" = var_mean_correction_2[1];  var_mean_correction_2 = None
        _output_to_half_0: "f16[1, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem, torch.float16);  getitem = None
        _output_to_half_1: "f16[1, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float16);  getitem_1 = None
        _output_to_half_2: "f16[1, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_2, torch.float16);  getitem_2 = None
        _output_to_half_3: "f16[1, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float16);  getitem_3 = None
        _output_to_half_4: "f16[1, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_4, torch.float16);  getitem_4 = None
        _output_to_half_5: "f16[1, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_5, torch.float16);  getitem_5 = None
        return (_output_to_half_0, _output_to_half_1, _output_to_half_2, _output_to_half_3, _output_to_half_4, _output_to_half_5)


def _default_make_inputs():
    return [
    torch.randn([128, 384, 14, 14], dtype=torch.float16, device='cuda'),
    torch.randn([128, 384, 14, 14], dtype=torch.float16, device='cuda'),
    torch.randn([128, 384, 14, 14], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
