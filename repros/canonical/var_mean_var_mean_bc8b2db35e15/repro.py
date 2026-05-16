"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g20
Pattern hash: bc8b2db35e15
Shape hash: fc9f5e06
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_33: "f16[1600, 768]", add_79: "f32[32, 50, 768]", addmm_81: "f16[2464, 512]", add_155: "f32[77, 32, 512]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[50, 32, 768]" = torch.ops.aten.reshape.default(addmm_33, [50, 32, 768]);  addmm_33 = None
        permute_default: "f16[32, 50, 768]" = torch.ops.aten.permute.default(reshape_default, [1, 0, 2]);  reshape_default = None
        add_tensor: "f32[32, 50, 768]" = torch.ops.aten.add.Tensor(add_79, permute_default);  add_79 = permute_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True);  add_tensor = None
        getitem: "f32[32, 50, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 50, 1]" = var_mean_correction[1];  var_mean_correction = None
        reshape_default_1: "f16[77, 32, 512]" = torch.ops.aten.reshape.default(addmm_81, [77, 32, 512]);  addmm_81 = None
        add_tensor_1: "f32[77, 32, 512]" = torch.ops.aten.add.Tensor(add_155, reshape_default_1);  add_155 = reshape_default_1 = None
        clone_default: "f32[77, 32, 512]" = torch.ops.aten.clone.default(add_tensor_1, memory_format = torch.contiguous_format);  add_tensor_1 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(clone_default, [2], correction = 0, keepdim = True);  clone_default = None
        getitem_2: "f32[77, 32, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[77, 32, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        return (getitem, getitem_1, getitem_2, getitem_3)


def _default_make_inputs():
    return [
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_155
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
