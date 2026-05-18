"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_inference
Pattern hash: 294676627a35
Shape hash: 6ba12287
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
    def forward(self, view_631: "f32[32, 7, 7, 1024]", getitem_170: "f32[32, 7, 7, 1]", getitem_169: "f32[32, 7, 7, 1]", arg347_1: "f32[1024]", arg348_1: "f32[1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, arg349_1: "f32[3072, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        sub_tensor: "f32[32, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(view_631, getitem_170);  view_631 = getitem_170 = None
        add_tensor: "f32[32, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_169, 1e-05);  getitem_169 = None
        rsqrt_default: "f32[32, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[32, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg347_1);  mul_tensor = arg347_1 = None
        add_tensor_1: "f32[32, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg348_1);  mul_tensor_1 = arg348_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        reshape_default: "f32[32, 1, 7, 1, 7, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  add_tensor_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_default: "f32[32, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 3, 2, 4, 5]);  reshape_default = None
        reshape_default_1: "f32[32, 7, 7, 1024]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        reshape_default_2: "f32[32, 49, 1024]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        reshape_default_3: "f32[1568, 1024]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f32[1024, 3072]" = torch.ops.aten.permute.default(arg349_1, [1, 0]);  arg349_1 = None
        return (reshape_default_3, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([32, 7, 7, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([32, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    [32, 1, 7, 1, 7, 1024],  # _shape_param_0
    [-1, 7, 7, 1024],  # _shape_param_1
    [-1, 49, 1024],  # _shape_param_2
    [1568, 1024],  # _shape_param_3
    torch.randn([3072, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
