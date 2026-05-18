"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_training
Pattern hash: 830b6fb095f1
Shape hash: e4505d3a
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
    def forward(self, bmm_140: "f32[8192, 49, 32]", _shape_param_0, bmm_142: "f32[8192, 32, 49]", _shape_param_1, bmm_143: "f32[8192, 49, 32]", _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, primals_8: "f32[384, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        reshape_default: "f32[2048, 4, 49, 32]" = torch.ops.aten.reshape.default(bmm_140, _shape_param_0);  bmm_140 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        reshape_default_1: "f32[2048, 4, 32, 49]" = torch.ops.aten.reshape.default(bmm_142, _shape_param_1);  bmm_142 = _shape_param_1 = None
        reshape_default_2: "f32[2048, 4, 49, 32]" = torch.ops.aten.reshape.default(bmm_143, _shape_param_2);  bmm_143 = _shape_param_2 = None
        permute_default: "f32[2048, 4, 49, 32]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_tensor: "f32[2048, 4, 49, 32]" = torch.ops.aten.mul.Tensor(reshape_default_2, 0.1767766952966369);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_default: "f32[6144, 4, 49, 32]" = torch.ops.aten.cat.default([mul_tensor, permute_default, reshape_default]);  mul_tensor = permute_default = reshape_default = None
        reshape_default_3: "f32[3, 2048, 4, 49, 32]" = torch.ops.aten.reshape.default(cat_default, _shape_param_3);  cat_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_1: "f32[2048, 49, 3, 4, 32]" = torch.ops.aten.permute.default(reshape_default_3, [1, 3, 0, 2, 4]);  reshape_default_3 = None
        clone_default: "f32[2048, 49, 3, 4, 32]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_4: "f32[2048, 49, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        reshape_default_5: "f32[100352, 384]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_2: "f32[128, 384]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_default_3: "f32[384, 128]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None
        return (reshape_default_5, permute_default_3)


def _default_make_inputs():
    return [
    torch.randn([8192, 49, 32], dtype=torch.float32, device='cuda'),
    [2048, 4, 49, 32],  # _shape_param_0
    torch.randn([8192, 32, 49], dtype=torch.float32, device='cuda'),
    [2048, 4, 32, 49],  # _shape_param_1
    torch.randn([8192, 49, 32], dtype=torch.float32, device='cuda'),
    [2048, 4, 49, 32],  # _shape_param_2
    [3, 2048, 4, 49, 32],  # _shape_param_3
    [2048, 49, 384],  # _shape_param_4
    [100352, 384],  # _shape_param_5
    torch.randn([384, 128], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
