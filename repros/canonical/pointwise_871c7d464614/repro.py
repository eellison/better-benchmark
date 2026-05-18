"""
Standalone repro captured via capture_hook.
Label: swin_t_inference
Pattern hash: 871c7d464614
Shape hash: 1bbed59f
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
    def forward(self, addmm_44: "f32[49, 2304]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        reshape_default: "f32[1, 49, 2304]" = torch.ops.aten.reshape.default(addmm_44, _shape_param_0);  addmm_44 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:180 in shifted_window_attention, code: qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        reshape_default_1: "f32[1, 49, 3, 24, 32]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[3, 1, 24, 49, 32]" = torch.ops.aten.permute.default(reshape_default_1, [2, 0, 3, 1, 4]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:181 in shifted_window_attention, code: q, k, v = qkv[0], qkv[1], qkv[2]
        select_int: "f32[1, 24, 49, 32]" = torch.ops.aten.select.int(permute_default, 0, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:188 in shifted_window_attention, code: q = q * (C // num_heads) ** -0.5
        mul_tensor: "f32[1, 24, 49, 32]" = torch.ops.aten.mul.Tensor(select_int, 0.1767766952966369);  select_int = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        expand_default: "f32[1, 24, 49, 32]" = torch.ops.aten.expand.default(mul_tensor, _shape_param_2);  mul_tensor = _shape_param_2 = None
        reshape_default_2: "f32[24, 49, 32]" = torch.ops.aten.reshape.default(expand_default, _shape_param_3);  expand_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:181 in shifted_window_attention, code: q, k, v = qkv[0], qkv[1], qkv[2]
        select_int_1: "f32[1, 24, 49, 32]" = torch.ops.aten.select.int(permute_default, 0, 1);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        permute_default_1: "f32[1, 24, 32, 49]" = torch.ops.aten.permute.default(select_int_1, [0, 1, 3, 2]);  select_int_1 = None
        expand_default_1: "f32[1, 24, 32, 49]" = torch.ops.aten.expand.default(permute_default_1, _shape_param_4);  permute_default_1 = _shape_param_4 = None
        reshape_default_3: "f32[24, 32, 49]" = torch.ops.aten.reshape.default(expand_default_1, _shape_param_5);  expand_default_1 = _shape_param_5 = None
        return (reshape_default_2, reshape_default_3)


def _default_make_inputs():
    return [
    torch.randn([49, 2304], dtype=torch.float32, device='cuda'),
    [1, 49, 2304],  # _shape_param_0
    [1, 49, 3, 24, 32],  # _shape_param_1
    [1, 24, 49, 32],  # _shape_param_2
    [24, 49, 32],  # _shape_param_3
    [1, 24, 32, 49],  # _shape_param_4
    [24, 32, 49],  # _shape_param_5
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
