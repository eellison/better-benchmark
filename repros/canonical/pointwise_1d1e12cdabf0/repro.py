"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g20
Pattern hash: 1d1e12cdabf0
Shape hash: a722b9ba
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
    def forward(self, getitem_147: "f16[32, 12, 50, 64]", arg143_1: "f32[768]", arg142_1: "f32[768, 768]", getitem_305: "f16[32, 8, 77, 64]", arg294_1: "f32[512]", arg293_1: "f32[512, 512]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        permute_default: "f16[50, 32, 12, 64]" = torch.ops.aten.permute.default(getitem_147, [2, 0, 1, 3]);  getitem_147 = None
        reshape_default: "f16[1600, 768]" = torch.ops.aten.reshape.default(permute_default, _shape_param_0);  permute_default = _shape_param_0 = None
        convert_element_type_default: "f16[768]" = torch.ops.prims.convert_element_type.default(arg143_1, torch.float16);  arg143_1 = None
        convert_element_type_default_1: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(arg142_1, torch.float16);  arg142_1 = None
        permute_default_1: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        permute_default_2: "f16[77, 32, 8, 64]" = torch.ops.aten.permute.default(getitem_305, [2, 0, 1, 3]);  getitem_305 = None
        reshape_default_1: "f16[2464, 512]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_1);  permute_default_2 = _shape_param_1 = None
        convert_element_type_default_2: "f16[512]" = torch.ops.prims.convert_element_type.default(arg294_1, torch.float16);  arg294_1 = None
        convert_element_type_default_3: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(arg293_1, torch.float16);  arg293_1 = None
        permute_default_3: "f16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_default_3, [1, 0]);  convert_element_type_default_3 = None
        return (reshape_default, convert_element_type_default, permute_default_1, reshape_default_1, convert_element_type_default_2, permute_default_3)


def _default_make_inputs():
    return [
    torch.randn(1228800, dtype=torch.float16, device='cuda').as_strided([32, 12, 50, 64], [768, 64, 24576, 1]),  # getitem_147
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # getitem_305
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    [1600, 768],  # _shape_param_0
    [2464, 512],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
