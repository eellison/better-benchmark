"""
Standalone repro captured via capture_hook.
Label: swin_t_inference
Pattern hash: 3d738a3a20af
Shape hash: 634844db
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
    def forward(self, addmm_39: "f32[196, 384]", _shape_param_0, add_106: "f32[1, 14, 14, 384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        reshape_default: "f32[1, 14, 14, 384]" = torch.ops.aten.reshape.default(addmm_39, _shape_param_0);  addmm_39 = _shape_param_0 = None
        add_tensor: "f32[1, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_106, reshape_default);  add_106 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:38 in _patch_merging_pad, code: x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        slice_tensor: "f32[1, 7, 14, 384]" = torch.ops.aten.slice.Tensor(add_tensor, 1, 0, 9223372036854775807, 2)
        slice_tensor_1: "f32[1, 7, 7, 384]" = torch.ops.aten.slice.Tensor(slice_tensor, 2, 0, 9223372036854775807, 2);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:39 in _patch_merging_pad, code: x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        slice_tensor_2: "f32[1, 7, 14, 384]" = torch.ops.aten.slice.Tensor(add_tensor, 1, 1, 9223372036854775807, 2)
        slice_tensor_3: "f32[1, 7, 7, 384]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 2, 0, 9223372036854775807, 2);  slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:40 in _patch_merging_pad, code: x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        slice_tensor_4: "f32[1, 7, 14, 384]" = torch.ops.aten.slice.Tensor(add_tensor, 1, 0, 9223372036854775807, 2)
        slice_tensor_5: "f32[1, 7, 7, 384]" = torch.ops.aten.slice.Tensor(slice_tensor_4, 2, 1, 9223372036854775807, 2);  slice_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:41 in _patch_merging_pad, code: x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        slice_tensor_6: "f32[1, 7, 14, 384]" = torch.ops.aten.slice.Tensor(add_tensor, 1, 1, 9223372036854775807, 2);  add_tensor = None
        slice_tensor_7: "f32[1, 7, 7, 384]" = torch.ops.aten.slice.Tensor(slice_tensor_6, 2, 1, 9223372036854775807, 2);  slice_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:42 in _patch_merging_pad, code: x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        cat_default: "f32[1, 7, 7, 1536]" = torch.ops.aten.cat.default([slice_tensor_1, slice_tensor_3, slice_tensor_5, slice_tensor_7], -1);  slice_tensor_1 = slice_tensor_3 = slice_tensor_5 = slice_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:84 in forward, code: x = self.norm(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(cat_default, [3], correction = 0, keepdim = True);  cat_default = None
        getitem: "f32[1, 7, 7, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 7, 7, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([196, 384], dtype=torch.float32, device='cuda'),
    [1, 14, 14, 384],  # _shape_param_0
    torch.randn([1, 14, 14, 384], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
