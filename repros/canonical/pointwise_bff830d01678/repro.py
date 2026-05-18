"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: bff830d01678
Shape hash: 646590f6
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
    def forward(self, getitem_42: "f32[4, 14, 14, 1]", add_97: "f32[4, 14, 14, 384]", getitem_43: "f32[4, 14, 14, 1]", primals_138: "f32[384]", primals_139: "f32[384]", fmod_8: "i64[14]", _shape_param_0, _shape_param_1, _shape_param_2, primals_142: "f32[1152, 384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        add_tensor: "f32[4, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_default: "f32[4, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 14, 14, 384]" = torch.ops.aten.sub.Tensor(add_97, getitem_43);  add_97 = getitem_43 = None
        mul_tensor: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_138);  mul_tensor = primals_138 = None
        add_tensor_1: "f32[4, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_139);  mul_tensor_1 = primals_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:167 in shifted_window_attention, code: x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        index_tensor: "f32[4, 14, 14, 384]" = torch.ops.aten.index.Tensor(add_tensor_1, [None, fmod_8]);  add_tensor_1 = None
        index_tensor_1: "f32[4, 14, 14, 384]" = torch.ops.aten.index.Tensor(index_tensor, [None, None, fmod_8]);  index_tensor = fmod_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:171 in shifted_window_attention, code: x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
        reshape_default: "f32[4, 2, 7, 2, 7, 384]" = torch.ops.aten.reshape.default(index_tensor_1, _shape_param_0);  index_tensor_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:172 in shifted_window_attention, code: x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C
        permute_default: "f32[4, 2, 2, 7, 7, 384]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 3, 2, 4, 5]);  reshape_default = None
        clone_default: "f32[4, 2, 2, 7, 7, 384]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[16, 49, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        reshape_default_2: "f32[784, 384]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
        return (reshape_default_2, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([4, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [14], dtype=torch.int64, device='cuda'),
    [4, 2, 7, 2, 7, 384],  # _shape_param_0
    [16, 49, 384],  # _shape_param_1
    [784, 384],  # _shape_param_2
    torch.randn([1152, 384], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
