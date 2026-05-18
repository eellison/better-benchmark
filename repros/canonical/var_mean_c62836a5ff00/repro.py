"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_training
Pattern hash: c62836a5ff00
Shape hash: 3b95a69a
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_48: "f32[1, 768, 1, 1]", convolution_40: "f32[32, 768, 7, 7]", getitem_49: "f32[1, 768, 1, 1]", primals_141: "f32[768]", primals_142: "f32[768]", primals_143: "f32[1, 768, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        add_tensor: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_default: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[32, 768, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_49);  convolution_40 = getitem_49 = None
        mul_tensor: "f32[32, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1);  primals_141 = None
        unsqueeze_default_1: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[32, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_142, -1);  primals_142 = None
        unsqueeze_default_3: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[32, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:462 in forward_features, code: x = self.pos_drop(x + self.pos_embed3)
        add_tensor_2: "f32[32, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_1, primals_143);  add_tensor_1 = primals_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_2, [0, 2, 3], correction = 0, keepdim = True);  add_tensor_2 = None
        getitem: "f32[1, 768, 1, 1]" = var_mean_correction[0]
        getitem_50: "f32[1, 768, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_50)


def _default_make_inputs():
    return [
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 768, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 7, 7], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
