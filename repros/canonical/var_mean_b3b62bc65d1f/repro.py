"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_inference
Pattern hash: b3b62bc65d1f
Shape hash: 9e41806b
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg221_1: "f32[1536, 768, 1, 1]", _shape_param_0, convolution_75: "f32[32, 768, 8, 8]", view_162: "f32[1, 768, 1152]", getitem_109: "f32[1, 768, 1]", getitem_108: "f32[1, 768, 1]", arg219_1: "f32[768, 1, 1, 1]", _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(arg221_1, _shape_param_0);  arg221_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [0, 2], correction = 0, keepdim = True);  reshape_default = None
        getitem: "f32[1, 1536, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 1536, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_tensor: "f32[32, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_75, 0.5)
        mul_tensor_1: "f32[32, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_75, 0.7071067811865476);  convolution_75 = None
        erf_default: "f32[32, 768, 8, 8]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[32, 768, 8, 8]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[32, 768, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_tensor_3: "f32[32, 768, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.7015043497085571);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_162, getitem_109);  view_162 = getitem_109 = None
        add_tensor_1: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_default: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_4: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_5: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg219_1, 0.02946278254943948);  arg219_1 = None
        reshape_default_1: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_5, [-1]);  mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_default: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, -1);  reshape_default_1 = None
        mul_tensor_6: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default);  mul_tensor_4 = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_2: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_1);  mul_tensor_6 = _shape_param_1 = None
        return (mul_tensor_3, reshape_default_2, getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([1536, 768, 1, 1], dtype=torch.float32, device='cuda'),
    [1, 1536, -1],  # _shape_param_0
    torch.randn([32, 768, 8, 8], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1152], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768, 1, 1, 1], dtype=torch.float32, device='cuda'),
    [768, 128, 3, 3],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
