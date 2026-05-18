"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s2_g20
Pattern hash: becd404a0962
Shape hash: 61d52dcd
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
    def forward(self, avg_pool2d_2: "f16[4, 512, 7, 7]", convolution_89: "f16[4, 32, 7, 7]", convolution_91: "f16[4, 32, 7, 7]", convolution_93: "f16[4, 32, 7, 7]", convolution_95: "f16[4, 32, 7, 7]", convolution_97: "f16[4, 32, 7, 7]", convolution_99: "f16[4, 32, 7, 7]", convolution_101: "f16[4, 32, 7, 7]"):
        # No stacktrace found for following nodes
        cat_default: "f16[4, 736, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101], 1);  avg_pool2d_2 = convolution_89 = convolution_91 = convolution_93 = convolution_95 = convolution_97 = convolution_99 = convolution_101 = None
        convert_element_type_default: "f32[4, 736, 7, 7]" = torch.ops.prims.convert_element_type.default(cat_default, torch.float32);  cat_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[1, 736, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 736, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([4, 512, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([4, 32, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([4, 32, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([4, 32, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([4, 32, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([4, 32, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([4, 32, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([4, 32, 7, 7], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
