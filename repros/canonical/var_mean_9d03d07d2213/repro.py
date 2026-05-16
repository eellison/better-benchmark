"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s2_g20
Pattern hash: 9d03d07d2213
Shape hash: 259740b0
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution_87: "f16[4, 512, 14, 14]"):
        # No stacktrace found for following nodes
        avg_pool2d_default: "f16[4, 512, 7, 7]" = torch.ops.aten.avg_pool2d.default(convolution_87, [2, 2], [2, 2]);  convolution_87 = None
        convert_element_type_default: "f32[4, 512, 7, 7]" = torch.ops.prims.convert_element_type.default(avg_pool2d_default, torch.float32);  avg_pool2d_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[1, 512, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 512, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([4, 512, 14, 14], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
