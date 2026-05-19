"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train
Pattern hash: 1ada32e9d93a
Shape hash: 4d544b47
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
    def forward(self, getitem_339: "f32[128, 128, 48, 48]", getitem_342: "f32[128, 128, 48, 48]", convolution_3: "f32[128, 128, 48, 48]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        add_tensor: "f32[128, 128, 48, 48]" = torch.ops.aten.add.Tensor(getitem_339, getitem_342);  getitem_339 = getitem_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_tensor: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(add_tensor, 1.0);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_tensor_1: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.7015043497085571);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_tensor_2: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_3, 0.7071067811865476)
        erf_default: "f32[128, 128, 48, 48]" = torch.ops.aten.erf.default(mul_tensor_2);  mul_tensor_2 = None
        add_tensor_1: "f32[128, 128, 48, 48]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_3: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(add_tensor_1, 0.5);  add_tensor_1 = None
        mul_tensor_4: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_3, convolution_3)
        mul_tensor_5: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_tensor_4, -0.5);  mul_tensor_4 = None
        exp_default: "f32[128, 128, 48, 48]" = torch.ops.aten.exp.default(mul_tensor_5);  mul_tensor_5 = None
        mul_tensor_6: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_7: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_3, mul_tensor_6);  convolution_3 = mul_tensor_6 = None
        add_tensor_2: "f32[128, 128, 48, 48]" = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_7);  mul_tensor_3 = mul_tensor_7 = None
        mul_tensor_8: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_2);  mul_tensor_1 = add_tensor_2 = None
        return mul_tensor_8


def _default_make_inputs():
    return [
    torch.randn(37748736, dtype=torch.float32, device='cuda').as_strided([128, 128, 48, 48], [294912, 1, 6144, 128]),  # getitem_339
    torch.randn(37748736, dtype=torch.float32, device='cuda').as_strided([128, 128, 48, 48], [294912, 1, 6144, 128]),  # getitem_342
    torch.randn(37748736, dtype=torch.float32, device='cuda').as_strided([128, 128, 48, 48], [294912, 1, 6144, 128]),  # convolution_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
