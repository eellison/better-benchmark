"""
Standalone repro captured via capture_hook.
Label: vit_b_16_training
Pattern hash: 2545588020b0
Shape hash: 5e98ed0f
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_144: "f32[4, 12, 197, 64]", _shape_param_0, getitem_143: "f32[4, 12, 197, 64]", _shape_param_1, getitem_142: "f32[4, 12, 197, 64]", _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, full_default_1: "f32[3, 197, 4, 768]", _shape_param_6, _shape_param_7, primals_9: "f32[2304, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:113 in forward, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        clone_default: "f32[4, 12, 197, 64]" = torch.ops.aten.clone.default(getitem_144, memory_format = torch.contiguous_format);  getitem_144 = None
        reshape_default: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        clone_default_1: "f32[4, 12, 197, 64]" = torch.ops.aten.clone.default(getitem_143, memory_format = torch.contiguous_format);  getitem_143 = None
        reshape_default_1: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_1);  clone_default_1 = _shape_param_1 = None
        clone_default_2: "f32[4, 12, 197, 64]" = torch.ops.aten.clone.default(getitem_142, memory_format = torch.contiguous_format);  getitem_142 = None
        reshape_default_2: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_2);  clone_default_2 = _shape_param_2 = None
        permute_default: "f32[197, 48, 64]" = torch.ops.aten.permute.default(reshape_default, [1, 0, 2]);  reshape_default = None
        clone_default_3: "f32[197, 48, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_3: "f32[197, 4, 768]" = torch.ops.aten.reshape.default(clone_default_3, _shape_param_3);  clone_default_3 = _shape_param_3 = None
        permute_default_1: "f32[197, 48, 64]" = torch.ops.aten.permute.default(reshape_default_1, [1, 0, 2]);  reshape_default_1 = None
        clone_default_4: "f32[197, 48, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_4: "f32[197, 4, 768]" = torch.ops.aten.reshape.default(clone_default_4, _shape_param_4);  clone_default_4 = _shape_param_4 = None
        permute_default_2: "f32[197, 48, 64]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0, 2]);  reshape_default_2 = None
        clone_default_5: "f32[197, 48, 64]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        reshape_default_5: "f32[197, 4, 768]" = torch.ops.aten.reshape.default(clone_default_5, _shape_param_5);  clone_default_5 = _shape_param_5 = None
        select_scatter_default: "f32[3, 197, 4, 768]" = torch.ops.aten.select_scatter.default(full_default_1, reshape_default_3, 0, 2);  reshape_default_3 = None
        select_scatter_default_1: "f32[3, 197, 4, 768]" = torch.ops.aten.select_scatter.default(full_default_1, reshape_default_4, 0, 1);  reshape_default_4 = None
        add_tensor: "f32[3, 197, 4, 768]" = torch.ops.aten.add.Tensor(select_scatter_default, select_scatter_default_1);  select_scatter_default = select_scatter_default_1 = None
        select_scatter_default_2: "f32[3, 197, 4, 768]" = torch.ops.aten.select_scatter.default(full_default_1, reshape_default_5, 0, 0);  full_default_1 = reshape_default_5 = None
        add_tensor_1: "f32[3, 197, 4, 768]" = torch.ops.aten.add.Tensor(add_tensor, select_scatter_default_2);  add_tensor = select_scatter_default_2 = None
        unsqueeze_default: "f32[3, 197, 4, 1, 768]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 3);  add_tensor_1 = None
        permute_default_3: "f32[1, 197, 4, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_default, [3, 1, 2, 0, 4]);  unsqueeze_default = None
        squeeze_dim: "f32[197, 4, 3, 768]" = torch.ops.aten.squeeze.dim(permute_default_3, 0);  permute_default_3 = None
        clone_default_6: "f32[197, 4, 3, 768]" = torch.ops.aten.clone.default(squeeze_dim, memory_format = torch.contiguous_format);  squeeze_dim = None
        reshape_default_6: "f32[197, 4, 2304]" = torch.ops.aten.reshape.default(clone_default_6, _shape_param_6);  clone_default_6 = _shape_param_6 = None
        reshape_default_7: "f32[788, 2304]" = torch.ops.aten.reshape.default(reshape_default_6, _shape_param_7);  reshape_default_6 = _shape_param_7 = None
        permute_default_4: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_default_5: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_default_4, [1, 0]);  permute_default_4 = None
        return (reshape_default_7, permute_default_5)


def _default_make_inputs():
    return [
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 12, 197, 64], [151296, 64, 768, 1]),  # getitem_144
    [48, 197, 64],  # _shape_param_0
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 12, 197, 64], [151296, 64, 768, 1]),  # getitem_143
    [48, 197, 64],  # _shape_param_1
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 12, 197, 64], [151296, 64, 768, 1]),  # getitem_142
    [48, 197, 64],  # _shape_param_2
    [197, 4, 768],  # _shape_param_3
    [197, 4, 768],  # _shape_param_4
    [197, 4, 768],  # _shape_param_5
    torch.randn([3, 197, 4, 768], dtype=torch.float32, device='cuda'),
    [197, 4, 2304],  # _shape_param_6
    [788, 2304],  # _shape_param_7
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
