"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g20
Pattern hash: c7f0384c2111
Shape hash: fc9f5e06
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
    def forward(self, addmm_32: "f16[1600, 768]", add_76: "f32[32, 50, 768]", addmm_79: "f16[2464, 512]", add_152: "f32[77, 32, 512]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[32, 50, 768]" = torch.ops.aten.reshape.default(addmm_32, [32, 50, 768]);  addmm_32 = None
        add_tensor: "f32[32, 50, 768]" = torch.ops.aten.add.Tensor(add_76, reshape_default);  add_76 = reshape_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True);  add_tensor = None
        getitem: "f32[32, 50, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 50, 1]" = var_mean_correction[1];  var_mean_correction = None
        reshape_default_1: "f16[77, 32, 512]" = torch.ops.aten.reshape.default(addmm_79, [77, 32, 512]);  addmm_79 = None
        add_tensor_1: "f32[77, 32, 512]" = torch.ops.aten.add.Tensor(add_152, reshape_default_1);  add_152 = reshape_default_1 = None
        clone_default: "f32[77, 32, 512]" = torch.ops.aten.clone.default(add_tensor_1, memory_format = torch.contiguous_format);  add_tensor_1 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(clone_default, [2], correction = 0, keepdim = True);  clone_default = None
        getitem_2: "f32[77, 32, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[77, 32, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        _output_to_half_0: "f16[32, 50, 1]" = torch.ops.prims.convert_element_type.default(getitem, torch.float16);  getitem = None
        _output_to_half_1: "f16[32, 50, 1]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float16);  getitem_1 = None
        _output_to_half_2: "f16[77, 32, 1]" = torch.ops.prims.convert_element_type.default(getitem_2, torch.float16);  getitem_2 = None
        _output_to_half_3: "f16[77, 32, 1]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float16);  getitem_3 = None
        return (_output_to_half_0, _output_to_half_1, _output_to_half_2, _output_to_half_3)


def _default_make_inputs():
    return [
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_152
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
