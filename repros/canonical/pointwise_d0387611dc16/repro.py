"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g53
Pattern hash: d0387611dc16
Shape hash: b02c5c44
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
    def forward(self, mm_94: "f16[2464, 512]", getitem_71: "f16[32, 12, 50, 64]", getitem_70: "f16[32, 12, 50, 64]", getitem_69: "f16[32, 12, 50, 64]", full_5: "f16[3, 50, 32, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # No stacktrace found for following nodes
        reshape_default: "f16[77, 32, 8, 64]" = torch.ops.aten.reshape.default(mm_94, _shape_param_0);  mm_94 = _shape_param_0 = None
        permute_default: "f16[32, 8, 77, 64]" = torch.ops.aten.permute.default(reshape_default, [1, 2, 0, 3]);  reshape_default = None
        reshape_default_1: "f16[384, 50, 64]" = torch.ops.aten.reshape.default(getitem_71, _shape_param_1);  getitem_71 = _shape_param_1 = None
        reshape_default_2: "f16[384, 50, 64]" = torch.ops.aten.reshape.default(getitem_70, _shape_param_2);  getitem_70 = _shape_param_2 = None
        reshape_default_3: "f16[384, 50, 64]" = torch.ops.aten.reshape.default(getitem_69, _shape_param_3);  getitem_69 = _shape_param_3 = None
        permute_default_1: "f16[50, 384, 64]" = torch.ops.aten.permute.default(reshape_default_1, [1, 0, 2]);  reshape_default_1 = None
        reshape_default_4: "f16[50, 32, 768]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_4);  permute_default_1 = _shape_param_4 = None
        permute_default_2: "f16[50, 384, 64]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0, 2]);  reshape_default_2 = None
        reshape_default_5: "f16[50, 32, 768]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_5);  permute_default_2 = _shape_param_5 = None
        permute_default_3: "f16[50, 384, 64]" = torch.ops.aten.permute.default(reshape_default_3, [1, 0, 2]);  reshape_default_3 = None
        reshape_default_6: "f16[50, 32, 768]" = torch.ops.aten.reshape.default(permute_default_3, _shape_param_6);  permute_default_3 = _shape_param_6 = None
        select_scatter_default: "f16[3, 50, 32, 768]" = torch.ops.aten.select_scatter.default(full_5, reshape_default_4, 0, 2);  reshape_default_4 = None
        select_scatter_default_1: "f16[3, 50, 32, 768]" = torch.ops.aten.select_scatter.default(full_5, reshape_default_5, 0, 1);  reshape_default_5 = None
        add_tensor: "f16[3, 50, 32, 768]" = torch.ops.aten.add.Tensor(select_scatter_default, select_scatter_default_1);  select_scatter_default = select_scatter_default_1 = None
        select_scatter_default_2: "f16[3, 50, 32, 768]" = torch.ops.aten.select_scatter.default(full_5, reshape_default_6, 0, 0);  full_5 = reshape_default_6 = None
        add_tensor_1: "f16[3, 50, 32, 768]" = torch.ops.aten.add.Tensor(add_tensor, select_scatter_default_2);  add_tensor = select_scatter_default_2 = None
        unsqueeze_default: "f16[3, 50, 32, 1, 768]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 3);  add_tensor_1 = None
        permute_default_4: "f16[1, 50, 32, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_default, [3, 1, 2, 0, 4]);  unsqueeze_default = None
        squeeze_dim: "f16[50, 32, 3, 768]" = torch.ops.aten.squeeze.dim(permute_default_4, 0);  permute_default_4 = None
        clone_default: "f16[50, 32, 3, 768]" = torch.ops.aten.clone.default(squeeze_dim, memory_format = torch.contiguous_format);  squeeze_dim = None
        reshape_default_7: "f16[50, 32, 2304]" = torch.ops.aten.reshape.default(clone_default, _shape_param_7);  clone_default = _shape_param_7 = None
        reshape_default_8: "f16[1600, 2304]" = torch.ops.aten.reshape.default(reshape_default_7, _shape_param_8);  reshape_default_7 = _shape_param_8 = None
        return (permute_default, reshape_default_8)


def _default_make_inputs():
    return [
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float16, device='cuda').as_strided([32, 12, 50, 64], [768, 64, 24576, 1]),  # getitem_71
    torch.randn(1228800, dtype=torch.float16, device='cuda').as_strided([32, 12, 50, 64], [768, 64, 24576, 1]),  # getitem_70
    torch.randn(1228800, dtype=torch.float16, device='cuda').as_strided([32, 12, 50, 64], [768, 64, 24576, 1]),  # getitem_69
    torch.randn([3, 50, 32, 768], dtype=torch.float16, device='cuda'),
    [77, 32, 8, 64],  # _shape_param_0
    [384, 50, 64],  # _shape_param_1
    [384, 50, 64],  # _shape_param_2
    [384, 50, 64],  # _shape_param_3
    [50, 32, 768],  # _shape_param_4
    [50, 32, 768],  # _shape_param_5
    [50, 32, 768],  # _shape_param_6
    [50, 32, 2304],  # _shape_param_7
    [1600, 2304],  # _shape_param_8
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
