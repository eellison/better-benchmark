"""
Standalone repro captured via capture_hook.
Label: timm_deit_base_distilled_patch16_224_train
Pattern hash: 9c4d4a65cd7b
Shape hash: 5ae49b96
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
_shapes_config = "(T([25344, 768], f32), T([768], f32), T([128, 198, 768], f32), T([1, 198, 768], f32), T([128, 198, 1], f32), T([128, 198, 1], f32), T([128, 198, 768], f32), S([128, 198, 768]), S([128, 768, 14, 14]))"

class Repro(torch.nn.Module):
    def forward(self, mm_98: "f32[25344, 768]", primals_7: "f32[768]", cat: "f32[128, 198, 768]", primals_4: "f32[1, 198, 768]", getitem_1: "f32[128, 198, 1]", rsqrt: "f32[128, 198, 1]", add_135: "f32[128, 198, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default: "f32[128, 198, 768]" = torch.ops.aten.reshape.default(mm_98, _shape_param_0);  mm_98 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(reshape_default, primals_7);  primals_7 = None
        mul_tensor_1: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[128, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:110 in _pos_embed, code: x = x + pos_embed
        add_tensor: "f32[128, 198, 768]" = torch.ops.aten.add.Tensor(cat, primals_4);  cat = primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_tensor: "f32[128, 198, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor_2: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_3: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[128, 198, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[128, 198, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[128, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_tensor_5: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_6: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor_2);  mul_tensor_2 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0, 1]);  reshape_default = None
        add_tensor_1: "f32[128, 198, 768]" = torch.ops.aten.add.Tensor(add_135, mul_tensor_5);  add_135 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:110 in _pos_embed, code: x = x + pos_embed
        sum_dim_int_list_4: "f32[1, 198, 768]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:105 in _pos_embed, code: x = torch.cat((
        slice_tensor: "f32[128, 1, 768]" = torch.ops.aten.slice.Tensor(add_tensor_1, 1, 0, 1)
        slice_tensor_1: "f32[128, 1, 768]" = torch.ops.aten.slice.Tensor(add_tensor_1, 1, 1, 2)
        slice_tensor_2: "f32[128, 196, 768]" = torch.ops.aten.slice.Tensor(add_tensor_1, 1, 2, 198);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:107 in _pos_embed, code: self.dist_token.expand(x.shape[0], -1, -1),
        sum_dim_int_list_5: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_tensor_1, [0], True);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:106 in _pos_embed, code: self.cls_token.expand(x.shape[0], -1, -1),
        sum_dim_int_list_6: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_tensor, [0], True);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        permute_default: "f32[128, 768, 196]" = torch.ops.aten.permute.default(slice_tensor_2, [0, 2, 1]);  slice_tensor_2 = None
        reshape_default_1: "f32[128, 768, 14, 14]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        sum_dim_int_list_7: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_1, [0, 2, 3]);  reshape_default_1 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, sum_dim_int_list_4, sum_dim_int_list_5, sum_dim_int_list_6, sum_dim_int_list_7)



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
