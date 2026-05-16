"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_inference
Pattern hash: e2e05b0d947e
Shape hash: 1aaaee06
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg36_1: "f32[256, 256, 1, 1]", _shape_param_0, convolution_9: "f32[32, 128, 1, 1]", view_12: "f32[1, 256, 128]", getitem_9: "f32[1, 256, 1]", getitem_8: "f32[1, 256, 1]", arg14_1: "f32[256, 1, 1, 1]", _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default: "f32[1, 256, 256]" = torch.ops.aten.reshape.default(arg36_1, _shape_param_0);  arg36_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [0, 2], correction = 0, keepdim = True);  reshape_default = None
        getitem: "f32[1, 256, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 256, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_default: "f32[32, 128, 1, 1]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_12, getitem_9);  view_12 = getitem_9 = None
        add_tensor: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_default: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_1: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg14_1, 0.08838834764831845);  arg14_1 = None
        reshape_default_1: "f32[256]" = torch.ops.aten.reshape.default(mul_tensor_1, [-1]);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_default: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, -1);  reshape_default_1 = None
        mul_tensor_2: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default);  mul_tensor = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_2: "f32[256, 128, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_1);  mul_tensor_2 = _shape_param_1 = None
        return (relu_default, reshape_default_2, getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([256, 256, 1, 1], dtype=torch.float32, device='cuda'),
    [1, 256, -1],  # _shape_param_0
    torch.randn([32, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256, 1, 1, 1], dtype=torch.float32, device='cuda'),
    [256, 128, 1, 1],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
