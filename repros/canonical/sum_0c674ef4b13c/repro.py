"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: 0c674ef4b13c
Shape hash: f4ac7cfe
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([32768, 49, 32], f32), T([32768, 32, 49], f32), T([32768, 49, 32], f32), S([8192, 4, 49, 32]), S([8192, 4, 32, 49]), S([8192, 4, 49, 32]), S([3, 8192, 4, 49, 32]), S([8192, 49, 384]), S([401408, 384]), S([384]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_140: "f32[32768, 49, 32]", bmm_142: "f32[32768, 32, 49]", bmm_143: "f32[32768, 49, 32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        reshape_default: "f32[8192, 4, 49, 32]" = torch.ops.aten.reshape.default(bmm_140, _shape_param_0);  bmm_140 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        reshape_default_1: "f32[8192, 4, 32, 49]" = torch.ops.aten.reshape.default(bmm_142, _shape_param_1);  bmm_142 = _shape_param_1 = None
        reshape_default_2: "f32[8192, 4, 49, 32]" = torch.ops.aten.reshape.default(bmm_143, _shape_param_2);  bmm_143 = _shape_param_2 = None
        permute_default: "f32[8192, 4, 49, 32]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_tensor: "f32[8192, 4, 49, 32]" = torch.ops.aten.mul.Tensor(reshape_default_2, 0.1767766952966369);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_default: "f32[24576, 4, 49, 32]" = torch.ops.aten.cat.default([mul_tensor, permute_default, reshape_default]);  mul_tensor = permute_default = reshape_default = None
        reshape_default_3: "f32[3, 8192, 4, 49, 32]" = torch.ops.aten.reshape.default(cat_default, _shape_param_3);  cat_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_1: "f32[8192, 49, 3, 4, 32]" = torch.ops.aten.permute.default(reshape_default_3, [1, 3, 0, 2, 4]);  reshape_default_3 = None
        clone_default: "f32[8192, 49, 3, 4, 32]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_4: "f32[8192, 49, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        reshape_default_5: "f32[401408, 384]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_2: "f32[384, 401408]" = torch.ops.aten.permute.default(reshape_default_5, [1, 0])
        sum_dim_int_list: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(reshape_default_5, [0], True);  reshape_default_5 = None
        reshape_default_6: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_6);  sum_dim_int_list = _shape_param_6 = None
        return (permute_default_2, reshape_default_6)



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
