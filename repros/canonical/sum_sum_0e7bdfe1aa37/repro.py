"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch16_siglip_256_train
Pattern hash: 0e7bdfe1aa37
Shape hash: 0a8651c2
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
_shapes_config = "(T([32768, 768], f32), T([768], f32), T([128, 768, 16, 16], f32, stride=(196608, 1, 12288, 768)), T([1, 256, 768], f32), T([128, 256, 1], f32), T([128, 256, 1], f32), T([128, 256, 768], f32), S([128, 256, 768]), S([128, 768, 256]), S([128, 768, 16, 16]))"

class Repro(torch.nn.Module):
    def forward(self, mm_105: "f32[32768, 768]", primals_5: "f32[768]", convolution: "f32[128, 768, 16, 16]", primals_4: "f32[1, 256, 768]", getitem_1: "f32[128, 256, 1]", rsqrt: "f32[128, 256, 1]", add_141: "f32[128, 256, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default: "f32[128, 256, 768]" = torch.ops.aten.reshape.default(mm_105, _shape_param_0);  mm_105 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(reshape_default, primals_5);  reshape_default = primals_5 = None
        mul_tensor_1: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[128, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        reshape_default_1: "f32[128, 768, 256]" = torch.ops.aten.reshape.default(convolution, _shape_param_1);  convolution = _shape_param_1 = None
        permute_default: "f32[128, 256, 768]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1073 in _pos_embed, code: x = x + pos_embed
        add_tensor: "f32[128, 256, 768]" = torch.ops.aten.add.Tensor(permute_default, primals_4);  permute_default = primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_tensor: "f32[128, 256, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor_2: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_3: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[128, 256, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[128, 256, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_tensor_5: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        add_tensor_1: "f32[128, 256, 768]" = torch.ops.aten.add.Tensor(add_141, mul_tensor_5);  add_141 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        permute_default_1: "f32[128, 768, 256]" = torch.ops.aten.permute.default(add_tensor_1, [0, 2, 1]);  add_tensor_1 = None
        reshape_default_2: "f32[128, 768, 16, 16]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_2);  permute_default_1 = _shape_param_2 = None
        return reshape_default_2



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
