"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_inference
Pattern hash: d300634a05a0
Shape hash: be1771f9
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg19_1: "f32[64, 64, 3, 3]", _shape_param_0, convolution_3: "f32[32, 128, 72, 72]", view_15: "f32[1, 64, 128]", getitem_11: "f32[1, 64, 1]", getitem_10: "f32[1, 64, 1]", arg17_1: "f32[64, 1, 1, 1]", _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default: "f32[1, 64, 576]" = torch.ops.aten.reshape.default(arg19_1, _shape_param_0);  arg19_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [0, 2], correction = 0, keepdim = True);  reshape_default = None
        getitem: "f32[1, 64, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 64, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_default: "f32[32, 128, 72, 72]" = torch.ops.aten.neg.default(convolution_3)
        exp_default: "f32[32, 128, 72, 72]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[32, 128, 72, 72]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[32, 128, 72, 72]" = torch.ops.aten.div.Tensor(convolution_3, add_tensor);  convolution_3 = add_tensor = None
        mul_tensor: "f32[32, 128, 72, 72]" = torch.ops.aten.mul.Tensor(div_tensor, 1.0);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_tensor: "f32[1, 64, 128]" = torch.ops.aten.sub.Tensor(view_15, getitem_11);  view_15 = getitem_11 = None
        add_tensor_1: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_default: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_1: "f32[1, 64, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_2: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg17_1, 0.1580497968320339);  arg17_1 = None
        reshape_default_1: "f32[64]" = torch.ops.aten.reshape.default(mul_tensor_2, [-1]);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_default: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, -1);  reshape_default_1 = None
        mul_tensor_3: "f32[1, 64, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default);  mul_tensor_1 = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        reshape_default_2: "f32[64, 128, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_1);  mul_tensor_3 = _shape_param_1 = None
        return (mul_tensor, reshape_default_2, getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([64, 64, 3, 3], dtype=torch.float32, device='cuda'),
    [1, 64, -1],  # _shape_param_0
    torch.randn([32, 128, 72, 72], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1, 1, 1], dtype=torch.float32, device='cuda'),
    [64, 128, 1, 1],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
