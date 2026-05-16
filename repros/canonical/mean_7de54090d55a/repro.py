"""
Standalone repro captured via capture_hook.
Label: efficientnet_b0_inference
Pattern hash: 7de54090d55a
Shape hash: b03d04e6
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg306_1: "f32[1280]", convolution_80: "f32[1, 1280, 7, 7]", arg307_1: "f32[1280]", arg308_1: "f32[1280]", arg309_1: "f32[1280]", _shape_param_0, arg310_1: "f32[1000, 1280]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:334 in _forward_impl, code: x = self.features(x)
        unsqueeze_default: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg306_1, -1);  arg306_1 = None
        unsqueeze_default_1: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[1, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_default_1);  convolution_80 = unsqueeze_default_1 = None
        add_tensor: "f32[1280]" = torch.ops.aten.add.Tensor(arg307_1, 1e-05);  arg307_1 = None
        sqrt_default: "f32[1280]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[1280]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[1280]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[1, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg308_1, -1);  arg308_1 = None
        unsqueeze_default_5: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[1, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg309_1, -1);  arg309_1 = None
        unsqueeze_default_7: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[1, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        neg_default: "f32[1, 1280, 7, 7]" = torch.ops.aten.neg.default(add_tensor_1)
        exp_default: "f32[1, 1280, 7, 7]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_2: "f32[1, 1280, 7, 7]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[1, 1280, 7, 7]" = torch.ops.aten.div.Tensor(add_tensor_1, add_tensor_2);  add_tensor_1 = add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:336 in _forward_impl, code: x = self.avgpool(x)
        mean_dim: "f32[1, 1280, 1, 1]" = torch.ops.aten.mean.dim(div_tensor, [-1, -2], True);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:337 in _forward_impl, code: x = torch.flatten(x, 1)
        reshape_default: "f32[1, 1280]" = torch.ops.aten.reshape.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:339 in _forward_impl, code: x = self.classifier(x)
        permute_default: "f32[1280, 1000]" = torch.ops.aten.permute.default(arg310_1, [1, 0]);  arg310_1 = None
        return (reshape_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn([1280], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1280, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([1280], dtype=torch.float32, device='cuda'),
    torch.randn([1280], dtype=torch.float32, device='cuda'),
    torch.randn([1280], dtype=torch.float32, device='cuda'),
    [1, 1280],  # _shape_param_0
    torch.randn([1000, 1280], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
