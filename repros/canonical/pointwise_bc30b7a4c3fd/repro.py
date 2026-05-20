"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_infer
Pattern hash: bc30b7a4c3fd
Shape hash: 540a56ff
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([768], f32), T([128, 768, 7, 7], f32, stride=(37632, 1, 5376, 768)), T([768], f32), T([768], f32), T([768], f32), T([1, 768, 7, 7], f32, stride=(37632, 1, 5376, 768)), T([768], f32), T([768], f32), T([768], f32), T([768], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg119_1: "f32[768]", convolution_40: "f32[128, 768, 7, 7]", arg120_1: "f32[768]", arg121_1: "f32[768]", arg122_1: "f32[768]", arg123_1: "f32[1, 768, 7, 7]", arg124_1: "f32[768]", arg125_1: "f32[768]", arg126_1: "f32[768]", arg127_1: "f32[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        unsqueeze_default: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_default_1: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 768, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_default_1);  convolution_40 = unsqueeze_default_1 = None
        add_tensor: "f32[768]" = torch.ops.aten.add.Tensor(arg120_1, 1e-05);  arg120_1 = None
        sqrt_default: "f32[768]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_default_5: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_default_7: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:462 in forward_features, code: x = self.pos_drop(x + self.pos_embed3)
        add_tensor_2: "f32[128, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_1, arg123_1);  add_tensor_1 = arg123_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_default_8: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_default_9: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        sub_tensor_1: "f32[128, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_tensor_2, unsqueeze_default_9);  add_tensor_2 = unsqueeze_default_9 = None
        add_tensor_3: "f32[768]" = torch.ops.aten.add.Tensor(arg125_1, 1e-05);  arg125_1 = None
        sqrt_default_1: "f32[768]" = torch.ops.aten.sqrt.default(add_tensor_3);  add_tensor_3 = None
        reciprocal_default_1: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_default_1);  sqrt_default_1 = None
        mul_tensor_3: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1);  reciprocal_default_1 = None
        unsqueeze_default_10: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_11: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        mul_tensor_4: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_11);  sub_tensor_1 = unsqueeze_default_11 = None
        unsqueeze_default_12: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_default_13: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_5: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_13);  mul_tensor_4 = unsqueeze_default_13 = None
        unsqueeze_default_14: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_default_15: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_4: "f32[128, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_15);  mul_tensor_5 = unsqueeze_default_15 = None
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
