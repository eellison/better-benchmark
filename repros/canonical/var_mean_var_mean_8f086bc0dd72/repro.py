"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: 8f086bc0dd72
Shape hash: d63d9d3d
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 768, 7, 7], f32, stride=(37632, 1, 5376, 768)), T([768], f32), T([768], f32), T([1, 768, 7, 7], f32, stride=(37632, 1, 5376, 768)), T([768], f32), T([768], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_40: "f32[128, 768, 7, 7]", primals_141: "f32[768]", primals_142: "f32[768]", primals_143: "f32[1, 768, 7, 7]", primals_147: "f32[768]", primals_148: "f32[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 768, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 768, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 768, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_1);  convolution_40 = getitem_1 = None
        mul_tensor: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1);  primals_141 = None
        unsqueeze_default_1: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_142, -1);  primals_142 = None
        unsqueeze_default_3: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[128, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:462 in forward_features, code: x = self.pos_drop(x + self.pos_embed3)
        add_tensor_2: "f32[128, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_1, primals_143);  add_tensor_1 = primals_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 768, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 768, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_3: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_default_1: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        sub_tensor_1: "f32[128, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_tensor_2, getitem_3);  add_tensor_2 = getitem_3 = None
        mul_tensor_2: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        unsqueeze_default_4: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_147, -1);  primals_147 = None
        unsqueeze_default_5: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_148, -1);  primals_148 = None
        unsqueeze_default_7: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_4: "f32[128, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None
        return add_tensor_4


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
