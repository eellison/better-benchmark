"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_train
Pattern hash: b03c1bd7c82d
Shape hash: 7c3a611f
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
_shapes_config = "(T([25216, 768], f32), T([768], f32), T([128, 197, 768], f32), T([128, 197, 1], f32), T([128, 197, 1], f32), T([128, 197, 768], f32), S([128, 197, 768]), S([128, 768, 14, 14]))"

class Repro(torch.nn.Module):
    def forward(self, mm_96: "f32[25216, 768]", primals_6: "f32[768]", cat: "f32[128, 197, 768]", getitem_1: "f32[128, 197, 1]", rsqrt: "f32[128, 197, 1]", add_132: "f32[128, 197, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        reshape_default: "f32[128, 197, 768]" = torch.ops.aten.reshape.default(mm_96, _shape_param_0);  mm_96 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(reshape_default, primals_6);  reshape_default = primals_6 = None
        mul_tensor_1: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        sub_tensor: "f32[128, 197, 768]" = torch.ops.aten.sub.Tensor(cat, getitem_1);  cat = getitem_1 = None
        mul_tensor_2: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_3: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[128, 197, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[128, 197, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[128, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_tensor_5: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        add_tensor: "f32[128, 197, 768]" = torch.ops.aten.add.Tensor(add_132, mul_tensor_5);  add_132 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:787 in forward_features, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        slice_tensor: "f32[128, 196, 768]" = torch.ops.aten.slice.Tensor(add_tensor, 1, 1, 197);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        permute_default: "f32[128, 768, 196]" = torch.ops.aten.permute.default(slice_tensor, [0, 2, 1]);  slice_tensor = None
        reshape_default_1: "f32[128, 768, 14, 14]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None
        return reshape_default_1



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
