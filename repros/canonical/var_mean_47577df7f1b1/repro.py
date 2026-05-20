"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_infer
Pattern hash: 47577df7f1b1
Shape hash: 0da4c91a
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([384, 64, 3, 3], f32, stride=(576, 1, 192, 64)), T([128, 384, 7, 7], f32, stride=(18816, 1, 2688, 384)), T([384, 1, 1, 1], f32), S([1, 384, 576]), S([384, 64, 3, 3]))"

class Repro(torch.nn.Module):
    def forward(self, arg207_1: "f32[384, 64, 3, 3]", convolution_75: "f32[128, 384, 7, 7]", arg208_1: "f32[384, 1, 1, 1]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default: "f32[384, 64, 3, 3]" = torch.ops.aten.clone.default(arg207_1, memory_format = torch.contiguous_format);  arg207_1 = None
        reshape_default: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [0, 2], correction = 0, keepdim = True)
        getitem: "f32[1, 384, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 384, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_default: "f32[128, 384, 7, 7]" = torch.ops.aten.neg.default(convolution_75)
        exp_default: "f32[128, 384, 7, 7]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[128, 384, 7, 7]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 384, 7, 7]" = torch.ops.aten.div.Tensor(convolution_75, add_tensor);  convolution_75 = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_tensor: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(reshape_default, getitem_1);  reshape_default = getitem_1 = None
        add_tensor_1: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
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
        return (div_tensor, reshape_default_2)


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
