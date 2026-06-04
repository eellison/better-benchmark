"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: 87fd4ec7f051
Shape hash: f0c1d821
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([401408, 128], f32), T([128], f32), T([128, 128, 56, 56], f32, stride=(401408, 1, 7168, 128)), T([128, 56, 56, 1], f32), T([128, 56, 56, 1], f32), T([128], f32), T([128], f32), T([128, 56, 56, 1], f32), T([128, 56, 56, 1], f32), T([128, 56, 56, 128], f32), S([8192, 49, 128]), S([8192, 7, 7, 128]), S([128, 8, 8, 7, 7, 128]), S([128, 56, 56, 128]))"

class Repro(torch.nn.Module):
    def forward(self, mm_201: "f32[401408, 128]", primals_6: "f32[128]", convolution: "f32[128, 128, 56, 56]", getitem_1: "f32[128, 56, 56, 1]", rsqrt: "f32[128, 56, 56, 1]", primals_4: "f32[128]", primals_5: "f32[128]", getitem_3: "f32[128, 56, 56, 1]", rsqrt_1: "f32[128, 56, 56, 1]", view_1390: "f32[128, 56, 56, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        reshape_default: "f32[8192, 49, 128]" = torch.ops.aten.reshape.default(mm_201, _shape_param_0);  mm_201 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        reshape_default_1: "f32[8192, 7, 7, 128]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        reshape_default_2: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 2, 4, 5]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_default: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_3: "f32[128, 56, 56, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(reshape_default_3, primals_6);  primals_6 = None
        mul_tensor_1: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, 128)
        sum_dim_int_list: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/format.py:68 in nchw_to, code: x = x.permute(0, 2, 3, 1)
        permute_default_1: "f32[128, 56, 56, 128]" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        sub_tensor: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(permute_default_1, getitem_1);  permute_default_1 = getitem_1 = None
        mul_tensor_2: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_3: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_2, primals_4)
        add_tensor: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_tensor_3, primals_5);  mul_tensor_3 = primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        sub_tensor_1: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_3);  add_tensor = getitem_3 = None
        mul_tensor_4: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_1);  sub_tensor_1 = None
        mul_tensor_5: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_4);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [3], True);  mul_tensor_5 = None
        mul_tensor_6: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_4, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_2: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_3: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_tensor_2, mul_tensor_6);  sub_tensor_2 = mul_tensor_6 = None
        div_tensor: "f32[128, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 128);  rsqrt_1 = None
        mul_tensor_7: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_3);  div_tensor = sub_tensor_3 = None
        mul_tensor_8: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(reshape_default_3, mul_tensor_4);  mul_tensor_4 = None
        sum_dim_int_list_2: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1, 2]);  mul_tensor_8 = None
        sum_dim_int_list_3: "f32[128]" = torch.ops.aten.sum.dim_IntList(reshape_default_3, [0, 1, 2]);  reshape_default_3 = None
        add_tensor_1: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(view_1390, mul_tensor_7);  view_1390 = mul_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        mul_tensor_9: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_4);  primals_4 = None
        mul_tensor_10: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_9, 128)
        sum_dim_int_list_4: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [3], True)
        mul_tensor_11: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_9, mul_tensor_2);  mul_tensor_9 = None
        sum_dim_int_list_5: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [3], True);  mul_tensor_11 = None
        mul_tensor_12: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_5);  sum_dim_int_list_5 = None
        sub_tensor_4: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_10, sum_dim_int_list_4);  mul_tensor_10 = sum_dim_int_list_4 = None
        sub_tensor_5: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_tensor_4, mul_tensor_12);  sub_tensor_4 = mul_tensor_12 = None
        div_tensor_1: "f32[128, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
        mul_tensor_13: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_tensor_1, sub_tensor_5);  div_tensor_1 = sub_tensor_5 = None
        mul_tensor_14: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_2);  mul_tensor_2 = None
        sum_dim_int_list_6: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1, 2]);  mul_tensor_14 = None
        sum_dim_int_list_7: "f32[128]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1, 2]);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/format.py:68 in nchw_to, code: x = x.permute(0, 2, 3, 1)
        permute_default_2: "f32[128, 128, 56, 56]" = torch.ops.aten.permute.default(mul_tensor_13, [0, 3, 1, 2]);  mul_tensor_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        sum_dim_int_list_8: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_default_2, [0, 2, 3]);  permute_default_2 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, sum_dim_int_list_6, sum_dim_int_list_7, sum_dim_int_list_8)

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
