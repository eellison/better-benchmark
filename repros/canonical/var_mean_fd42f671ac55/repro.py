"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_inference
Pattern hash: fd42f671ac55
Shape hash: ba4f400d
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg210_1: "f32[1536, 384, 1, 1]", _shape_param_0, convolution_75: "f32[32, 384, 9, 9]", view_162: "f32[1, 384, 576]", getitem_109: "f32[1, 384, 1]", getitem_108: "f32[1, 384, 1]", arg208_1: "f32[384, 1, 1, 1]", _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(arg210_1, _shape_param_0);  arg210_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [0, 2], correction = 0, keepdim = True);  reshape_default = None
        getitem: "f32[1, 1536, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 1536, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_default: "f32[32, 384, 9, 9]" = torch.ops.aten.neg.default(convolution_75)
        exp_default: "f32[32, 384, 9, 9]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[32, 384, 9, 9]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[32, 384, 9, 9]" = torch.ops.aten.div.Tensor(convolution_75, add_tensor);  convolution_75 = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_tensor: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_162, getitem_109);  view_162 = getitem_109 = None
        add_tensor_1: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_default: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_1: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg208_1, 0.07450538873672485);  arg208_1 = None
        reshape_default_1: "f32[384]" = torch.ops.aten.reshape.default(mul_tensor_1, [-1]);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_default: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, -1);  reshape_default_1 = None
        mul_tensor_2: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default);  mul_tensor = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        reshape_default_2: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_1);  mul_tensor_2 = _shape_param_1 = None
        return (div_tensor, reshape_default_2, getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([1536, 384, 1, 1], dtype=torch.float32, device='cuda'),
    [1, 1536, -1],  # _shape_param_0
    torch.randn([32, 384, 9, 9], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 576], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384, 1, 1, 1], dtype=torch.float32, device='cuda'),
    [384, 64, 3, 3],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
