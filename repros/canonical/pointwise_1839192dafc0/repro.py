"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g53
Pattern hash: 1839192dafc0
Shape hash: c1e2be3d
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_6: "f16[2464, 512]", getitem_38: "f16[32, 12, 50, 64]", getitem_37: "f16[32, 12, 50, 64]", getitem_36: "f16[32, 12, 50, 64]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[77, 32, 8, 64]" = torch.ops.aten.reshape.default(mm_6, [77, 32, 8, 64]);  mm_6 = None
        permute_default: "f16[32, 8, 77, 64]" = torch.ops.aten.permute.default(reshape_default, [1, 2, 0, 3]);  reshape_default = None
        reshape_default_1: "f16[384, 50, 64]" = torch.ops.aten.reshape.default(getitem_38, [384, 50, 64]);  getitem_38 = None
        reshape_default_2: "f16[384, 50, 64]" = torch.ops.aten.reshape.default(getitem_37, [384, 50, 64]);  getitem_37 = None
        reshape_default_3: "f16[384, 50, 64]" = torch.ops.aten.reshape.default(getitem_36, [384, 50, 64]);  getitem_36 = None
        permute_default_1: "f16[50, 384, 64]" = torch.ops.aten.permute.default(reshape_default_1, [1, 0, 2]);  reshape_default_1 = None
        reshape_default_4: "f16[50, 32, 768]" = torch.ops.aten.reshape.default(permute_default_1, [50, 32, 768]);  permute_default_1 = None
        permute_default_2: "f16[50, 384, 64]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0, 2]);  reshape_default_2 = None
        reshape_default_5: "f16[50, 32, 768]" = torch.ops.aten.reshape.default(permute_default_2, [50, 32, 768]);  permute_default_2 = None
        permute_default_3: "f16[50, 384, 64]" = torch.ops.aten.permute.default(reshape_default_3, [1, 0, 2]);  reshape_default_3 = None
        reshape_default_6: "f16[50, 32, 768]" = torch.ops.aten.reshape.default(permute_default_3, [50, 32, 768]);  permute_default_3 = None
        full_default: "f16[3, 50, 32, 768]" = torch.ops.aten.full.default([3, 50, 32, 768], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default: "f16[3, 50, 32, 768]" = torch.ops.aten.select_scatter.default(full_default, reshape_default_4, 0, 2);  reshape_default_4 = None
        select_scatter_default_1: "f16[3, 50, 32, 768]" = torch.ops.aten.select_scatter.default(full_default, reshape_default_5, 0, 1);  reshape_default_5 = None
        add_tensor: "f16[3, 50, 32, 768]" = torch.ops.aten.add.Tensor(select_scatter_default, select_scatter_default_1);  select_scatter_default = select_scatter_default_1 = None
        select_scatter_default_2: "f16[3, 50, 32, 768]" = torch.ops.aten.select_scatter.default(full_default, reshape_default_6, 0, 0);  full_default = reshape_default_6 = None
        add_tensor_1: "f16[3, 50, 32, 768]" = torch.ops.aten.add.Tensor(add_tensor, select_scatter_default_2);  add_tensor = select_scatter_default_2 = None
        unsqueeze_default: "f16[3, 50, 32, 1, 768]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 3);  add_tensor_1 = None
        permute_default_4: "f16[1, 50, 32, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_default, [3, 1, 2, 0, 4]);  unsqueeze_default = None
        squeeze_dim: "f16[50, 32, 3, 768]" = torch.ops.aten.squeeze.dim(permute_default_4, 0);  permute_default_4 = None
        clone_default: "f16[50, 32, 3, 768]" = torch.ops.aten.clone.default(squeeze_dim, memory_format = torch.contiguous_format);  squeeze_dim = None
        reshape_default_7: "f16[50, 32, 2304]" = torch.ops.aten.reshape.default(clone_default, [50, 32, 2304]);  clone_default = None
        reshape_default_8: "f16[1600, 2304]" = torch.ops.aten.reshape.default(reshape_default_7, [1600, 2304]);  reshape_default_7 = None
        return (permute_default, reshape_default_8)


def _default_make_inputs():
    return [
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float16, device='cuda').as_strided([32, 12, 50, 64], [768, 64, 24576, 1]),  # getitem_38
    torch.randn(1228800, dtype=torch.float16, device='cuda').as_strided([32, 12, 50, 64], [768, 64, 24576, 1]),  # getitem_37
    torch.randn(1228800, dtype=torch.float16, device='cuda').as_strided([32, 12, 50, 64], [768, 64, 24576, 1]),  # getitem_36
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
