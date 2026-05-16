"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_inference
Pattern hash: 29d33ca96da0
Shape hash: a9c83d13
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg4_1: "f32[32, 16, 3, 3]", _shape_param_0, arg0_1: "f32[32, 3, 256, 256]", view: "f32[1, 16, 27]", getitem_1: "f32[1, 16, 1]", getitem: "f32[1, 16, 1]", arg2_1: "f32[16, 1, 1, 1]", _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default: "f32[1, 32, 144]" = torch.ops.aten.reshape.default(arg4_1, _shape_param_0);  arg4_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [0, 2], correction = 0, keepdim = True);  reshape_default = None
        getitem: "f32[1, 32, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 32, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[32, 3, 257, 257]" = torch.ops.aten.constant_pad_nd.default(arg0_1, [0, 1, 0, 1], 0.0);  arg0_1 = None
        getitem_2 = getitem_1

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view, getitem_2);  view = getitem_2 = None
        getitem_3 = getitem
        add_tensor: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_3, 1e-05);  getitem_3 = None
        rsqrt_default: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_1: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg2_1, 0.19245008972987526);  arg2_1 = None
        reshape_default_1: "f32[16]" = torch.ops.aten.reshape.default(mul_tensor_1, [-1]);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_default: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, -1);  reshape_default_1 = None
        mul_tensor_2: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default);  mul_tensor = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_2: "f32[16, 3, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_1);  mul_tensor_2 = _shape_param_1 = None
        return (constant_pad_nd_default, reshape_default_2, getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([32, 16, 3, 3], dtype=torch.float32, device='cuda'),
    [1, 32, -1],  # _shape_param_0
    torch.randn([32, 3, 256, 256], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 27], dtype=torch.float32, device='cuda'),
    torch.tensor(1),  # getitem_2 (unknown shape)
    torch.tensor(1),  # getitem_3 (unknown shape)
    torch.randn([16, 1, 1, 1], dtype=torch.float32, device='cuda'),
    [16, 3, 3, 3],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
