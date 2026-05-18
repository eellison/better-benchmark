"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g20
Pattern hash: e188b80db662
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
    def forward(self, addmm_35: "f16[1600, 768]", add_83: "f32[32, 50, 768]", addmm_83: "f16[2464, 512]", add_158: "f32[77, 32, 512]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        reshape_default: "f16[32, 50, 768]" = torch.ops.aten.reshape.default(addmm_35, _shape_param_0);  addmm_35 = _shape_param_0 = None
        add_tensor: "f32[32, 50, 768]" = torch.ops.aten.add.Tensor(add_83, reshape_default);  add_83 = reshape_default = None
        select_int: "f32[32, 768]" = torch.ops.aten.select.int(add_tensor, 1, 0);  add_tensor = None
        clone_default: "f32[32, 768]" = torch.ops.aten.clone.default(select_int, memory_format = torch.contiguous_format);  select_int = None
        var_mean_correction = torch.ops.aten.var_mean.correction(clone_default, [1], correction = 0, keepdim = True);  clone_default = None
        getitem: "f32[32, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 1]" = var_mean_correction[1];  var_mean_correction = None
        reshape_default_1: "f16[77, 32, 512]" = torch.ops.aten.reshape.default(addmm_83, _shape_param_1);  addmm_83 = _shape_param_1 = None
        add_tensor_1: "f32[77, 32, 512]" = torch.ops.aten.add.Tensor(add_158, reshape_default_1);  add_158 = reshape_default_1 = None
        permute_default: "f32[32, 77, 512]" = torch.ops.aten.permute.default(add_tensor_1, [1, 0, 2]);  add_tensor_1 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(permute_default, [2], correction = 0, keepdim = True);  permute_default = None
        getitem_2: "f32[32, 77, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[32, 77, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        _output_to_half_0: "f16[32, 1]" = torch.ops.prims.convert_element_type.default(getitem, torch.float16);  getitem = None
        _output_to_half_1: "f16[32, 1]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float16);  getitem_1 = None
        _output_to_half_2: "f16[32, 77, 1]" = torch.ops.prims.convert_element_type.default(getitem_2, torch.float16);  getitem_2 = None
        _output_to_half_3: "f16[32, 77, 1]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float16);  getitem_3 = None
        return (_output_to_half_0, _output_to_half_1, _output_to_half_2, _output_to_half_3)


def _default_make_inputs():
    return [
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_158,
    [32, 50, 768],  # _shape_param_0
    [77, 32, 512],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
