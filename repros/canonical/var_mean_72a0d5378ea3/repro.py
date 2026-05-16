"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s6_g10
Pattern hash: 72a0d5378ea3
Shape hash: 617d24de
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution_31: "bf16[64, 240, 8, 8]"):
        # No stacktrace found for following nodes
        reshape_default: "bf16[61440, 2, 4, 2]" = torch.ops.aten.reshape.default(convolution_31, [61440, 2, 4, 2]);  convolution_31 = None
        permute_default: "bf16[61440, 4, 2, 2]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "bf16[61440, 4, 2, 2]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "bf16[64, 240, 16, 4]" = torch.ops.aten.reshape.default(clone_default, [64, 240, 16, 4]);  clone_default = None
        permute_default_1: "bf16[64, 4, 16, 240]" = torch.ops.aten.permute.default(reshape_default_1, [0, 3, 2, 1]);  reshape_default_1 = None
        clone_default_1: "bf16[64, 4, 16, 240]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_2: "bf16[256, 16, 240]" = torch.ops.aten.reshape.default(clone_default_1, [256, 16, 240]);  clone_default_1 = None
        convert_element_type_default: "f32[256, 16, 240]" = torch.ops.prims.convert_element_type.default(reshape_default_2, torch.float32);  reshape_default_2 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[256, 16, 1]" = var_mean_correction[0]
        getitem_1: "f32[256, 16, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([64, 240, 8, 8], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
