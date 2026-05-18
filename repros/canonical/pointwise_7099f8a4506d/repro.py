"""
Standalone repro captured via capture_hook.
Label: swin_t_inference
Pattern hash: 7099f8a4506d
Shape hash: 25e39739
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
    def forward(self, add_120: "f32[1, 7, 7, 768]", getitem_53: "f32[1, 7, 7, 1]", getitem_52: "f32[1, 7, 7, 1]", arg168_1: "f32[768]", arg169_1: "f32[768]", _shape_param_0, _shape_param_1, _shape_param_2, arg172_1: "f32[2304, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        sub_tensor: "f32[1, 7, 7, 768]" = torch.ops.aten.sub.Tensor(add_120, getitem_53);  add_120 = getitem_53 = None
        add_tensor: "f32[1, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_default: "f32[1, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 7, 7, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 7, 7, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg168_1);  mul_tensor = arg168_1 = None
        add_tensor_1: "f32[1, 7, 7, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg169_1);  mul_tensor_1 = arg169_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:171 in shifted_window_attention, code: x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
        reshape_default: "f32[1, 1, 7, 1, 7, 768]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  add_tensor_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:172 in shifted_window_attention, code: x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C
        permute_default: "f32[1, 1, 1, 7, 7, 768]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 3, 2, 4, 5]);  reshape_default = None
        reshape_default_1: "f32[1, 49, 768]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        reshape_default_2: "f32[49, 768]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[768, 2304]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        return (reshape_default_2, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([1, 7, 7, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    [1, 1, 7, 1, 7, 768],  # _shape_param_0
    [1, 49, 768],  # _shape_param_1
    [49, 768],  # _shape_param_2
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
