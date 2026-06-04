"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_train
Pattern hash: fbf523037907
Shape hash: 2125e5dc
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 768, 14, 14], f32, stride=(150528, 1, 10752, 768)), T([1, 1, 768], f32), T([768], f32), T([768], f32), S([128, 768, 196]), S([128, -1, -1]), S([25216, 768]))"

class Repro(torch.nn.Module):
    def forward(self, convolution: "f32[128, 768, 14, 14]", primals_4: "f32[1, 1, 768]", primals_6: "f32[768]", primals_7: "f32[768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        reshape_default: "f32[128, 768, 196]" = torch.ops.aten.reshape.default(convolution, _shape_param_0);  convolution = _shape_param_0 = None
        permute_default: "f32[128, 196, 768]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:787 in forward_features, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        expand_default: "f32[128, 1, 768]" = torch.ops.aten.expand.default(primals_4, _shape_param_1);  primals_4 = _shape_param_1 = None
        cat_default: "f32[128, 197, 768]" = torch.ops.aten.cat.default([expand_default, permute_default], 1);  expand_default = permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_correction = torch.ops.aten.var_mean.correction(cat_default, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 197, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 197, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[128, 197, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 197, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 197, 768]" = torch.ops.aten.sub.Tensor(cat_default, getitem_1);  cat_default = getitem_1 = None
        mul_tensor: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_6);  mul_tensor = primals_6 = None
        add_tensor_1: "f32[128, 197, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_7);  mul_tensor_1 = primals_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        reshape_default_1: "f32[25216, 768]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_2);  add_tensor_1 = _shape_param_2 = None
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
