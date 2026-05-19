"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: ea2e2ccbae34
Shape hash: 76b64da2
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
    def forward(self, convolution_2: "f32[128, 384, 28, 28]", getitem_239: "f32[128, 384, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        mul_tensor: "f32[128, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_2, 0.7071067811865476)
        erf_default: "f32[128, 384, 28, 28]" = torch.ops.aten.erf.default(mul_tensor);  mul_tensor = None
        add_tensor: "f32[128, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_1: "f32[128, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_2: "f32[128, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_2, convolution_2)
        mul_tensor_3: "f32[128, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_2, -0.5);  mul_tensor_2 = None
        exp_default: "f32[128, 384, 28, 28]" = torch.ops.aten.exp.default(mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_5: "f32[128, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_2, mul_tensor_4);  convolution_2 = mul_tensor_4 = None
        add_tensor_1: "f32[128, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_5);  mul_tensor_1 = mul_tensor_5 = None
        mul_tensor_6: "f32[128, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_239, add_tensor_1);  getitem_239 = add_tensor_1 = None
        return mul_tensor_6


def _default_make_inputs():
    return [
    torch.randn(38535168, dtype=torch.float32, device='cuda').as_strided([128, 384, 28, 28], [301056, 1, 10752, 384]),  # convolution_2
    torch.randn(38535168, dtype=torch.float32, device='cuda').as_strided([128, 384, 28, 28], [301056, 1, 10752, 384]),  # getitem_239
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
