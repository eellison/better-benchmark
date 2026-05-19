"""
Standalone repro captured via capture_hook.
Label: timm_deit_base_distilled_patch16_224_infer
Pattern hash: 9c03b27495aa
Shape hash: 92da1ceb
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg4_1: "f32[1, 1, 768]", arg5_1: "f32[1, 1, 768]", convolution: "f32[128, 768, 14, 14]", arg3_1: "f32[1, 198, 768]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg8_1: "f32[2304, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:106 in _pos_embed, code: self.cls_token.expand(x.shape[0], -1, -1),
        expand_default: "f32[128, 1, 768]" = torch.ops.aten.expand.default(arg4_1, _shape_param_0);  arg4_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:107 in _pos_embed, code: self.dist_token.expand(x.shape[0], -1, -1),
        expand_default_1: "f32[128, 1, 768]" = torch.ops.aten.expand.default(arg5_1, _shape_param_1);  arg5_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        reshape_default: "f32[128, 768, 196]" = torch.ops.aten.reshape.default(convolution, _shape_param_2);  convolution = _shape_param_2 = None
        permute_default: "f32[128, 196, 768]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:105 in _pos_embed, code: x = torch.cat((
        cat_default: "f32[128, 198, 768]" = torch.ops.aten.cat.default([expand_default, expand_default_1, permute_default], 1);  expand_default = expand_default_1 = permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:110 in _pos_embed, code: x = x + pos_embed
        add_tensor: "f32[128, 198, 768]" = torch.ops.aten.add.Tensor(cat_default, arg3_1);  cat_default = arg3_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 198, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 198, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[128, 198, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[128, 198, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 198, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg6_1);  mul_tensor = arg6_1 = None
        add_tensor_2: "f32[128, 198, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg7_1);  mul_tensor_1 = arg7_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default_1: "f32[25344, 768]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_3);  add_tensor_2 = _shape_param_3 = None
        permute_default_1: "f32[768, 2304]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        return (reshape_default_1, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([1, 1, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1, 768], dtype=torch.float32, device='cuda'),
    torch.randn(19267584, dtype=torch.float32, device='cuda').as_strided([128, 768, 14, 14], [150528, 1, 10752, 768]),  # convolution
    torch.randn([1, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    [128, -1, -1],  # _shape_param_0
    [128, -1, -1],  # _shape_param_1
    [128, 768, 196],  # _shape_param_2
    [25344, 768],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
