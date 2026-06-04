"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_infer
Pattern hash: 2e61b19a52bc
Shape hash: 6a2dc974
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 240, 8, 8], f32, stride=(15360, 1, 1920, 240)), T([240], f32), T([240], f32), S([122880, 2, 4, 2]), S([128, 240, 16, 4]), S([512, 16, 240]), S([8192, 240]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_31: "f32[128, 240, 8, 8]", arg225_1: "f32[240]", arg226_1: "f32[240]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:258 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        clone_default: "f32[128, 240, 8, 8]" = torch.ops.aten.clone.default(convolution_31, memory_format = torch.contiguous_format);  convolution_31 = None
        reshape_default: "f32[122880, 2, 4, 2]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        permute_default: "f32[122880, 4, 2, 2]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        clone_default_1: "f32[122880, 4, 2, 2]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[128, 240, 16, 4]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_1);  clone_default_1 = _shape_param_1 = None
        permute_default_1: "f32[128, 4, 16, 240]" = torch.ops.aten.permute.default(reshape_default_1, [0, 3, 2, 1]);  reshape_default_1 = None
        clone_default_2: "f32[128, 4, 16, 240]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_2: "f32[512, 16, 240]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_2);  clone_default_2 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default_2, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 16, 1]" = var_mean_correction[0]
        getitem_1: "f32[512, 16, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(reshape_default_2, getitem_1);  reshape_default_2 = getitem_1 = None
        add_tensor: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul_tensor, arg225_1);  mul_tensor = arg225_1 = None
        add_tensor_1: "f32[512, 16, 240]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg226_1);  mul_tensor_1 = arg226_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default_3: "f32[8192, 240]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_3);  add_tensor_1 = _shape_param_3 = None
        return reshape_default_3

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
