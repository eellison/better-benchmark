"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_inference
Pattern hash: 9e590341f685
Shape hash: 14a19209
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
    def forward(self, addmm_85: "f32[6272, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, view_573: "f32[32, 14, 14, 512]", _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        reshape_default: "f32[128, 49, 512]" = torch.ops.aten.reshape.default(addmm_85, _shape_param_0);  addmm_85 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        reshape_default_1: "f32[128, 7, 7, 512]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        reshape_default_2: "f32[32, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_default: "f32[32, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 2, 4, 5]);  reshape_default_2 = None
        clone_default: "f32[32, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_3: "f32[32, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_default: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[14]" = torch.ops.aten.add.Tensor(iota_default, 11);  iota_default = None
        fmod_scalar: "i64[14]" = torch.ops.aten.fmod.Scalar(add_tensor, 14);  add_tensor = None
        index_tensor: "f32[32, 14, 14, 512]" = torch.ops.aten.index.Tensor(reshape_default_3, [None, fmod_scalar]);  reshape_default_3 = fmod_scalar = None
        iota_default_1: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_1: "i64[14]" = torch.ops.aten.add.Tensor(iota_default_1, 11);  iota_default_1 = None
        fmod_scalar_1: "i64[14]" = torch.ops.aten.fmod.Scalar(add_tensor_1, 14);  add_tensor_1 = None
        index_tensor_1: "f32[32, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_tensor, [None, None, fmod_scalar_1]);  index_tensor = fmod_scalar_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_tensor_2: "f32[32, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_573, index_tensor_1);  view_573 = index_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        reshape_default_4: "f32[32, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_4);  add_tensor_2 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default_4, [2], correction = 0, keepdim = True);  reshape_default_4 = None
        getitem: "f32[32, 196, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 196, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([6272, 512], dtype=torch.float32, device='cuda'),
    [128, 49, 512],  # _shape_param_0
    [-1, 7, 7, 512],  # _shape_param_1
    [-1, 2, 2, 7, 7, 512],  # _shape_param_2
    [-1, 14, 14, 512],  # _shape_param_3
    torch.randn([32, 14, 14, 512], dtype=torch.float32, device='cuda'),
    [32, -1, 512],  # _shape_param_4
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
