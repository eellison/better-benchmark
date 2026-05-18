"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s2_g20
Pattern hash: 5bdf00226e93
Shape hash: 5b0f1539
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
    def forward(self, getitem_181: "f16[8, 12, 198, 64]", arg143_1: "f32[768]", arg142_1: "f32[768, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        permute_default: "f16[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3]);  getitem_181 = None
        reshape_default: "f16[8, 198, 768]" = torch.ops.aten.reshape.default(permute_default, _shape_param_0);  permute_default = _shape_param_0 = None
        convert_element_type_default: "f16[768]" = torch.ops.prims.convert_element_type.default(arg143_1, torch.float16);  arg143_1 = None
        convert_element_type_default_1: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(arg142_1, torch.float16);  arg142_1 = None
        reshape_default_1: "f16[1584, 768]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default_1: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        return (convert_element_type_default, reshape_default_1, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn(1216512, dtype=torch.float16, device='cuda').as_strided([8, 12, 198, 64], [152064, 64, 768, 1]),  # getitem_181
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [8, 198, 768],  # _shape_param_0
    [1584, 768],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
