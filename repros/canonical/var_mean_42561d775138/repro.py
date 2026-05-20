"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train
Pattern hash: 42561d775138
Shape hash: 9e234aa7
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 3, 192, 192], f32, stride=(110592, 1, 576, 3)), T([16, 3, 3, 3], f32, stride=(27, 1, 9, 3)), T([16, 1, 1, 1], f32), S([1, 16, 27]), S([16, 3, 3, 3]))"

class Repro(torch.nn.Module):
    def forward(self, primals_1: "f32[128, 3, 192, 192]", primals_2: "f32[16, 3, 3, 3]", primals_3: "f32[16, 1, 1, 1]", _shape_param_0, _shape_param_1):
        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[128, 3, 193, 193]" = torch.ops.aten.constant_pad_nd.default(primals_1, [0, 1, 0, 1], 0.0);  primals_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default: "f32[16, 3, 3, 3]" = torch.ops.aten.clone.default(primals_2, memory_format = torch.contiguous_format);  primals_2 = None
        reshape_default: "f32[1, 16, 27]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_3, 0.19245008972987526);  primals_3 = None
        reshape_default_1: "f32[16]" = torch.ops.aten.reshape.default(mul_tensor, [-1]);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [0, 2], correction = 0, keepdim = True)
        getitem: "f32[1, 16, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 16, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(reshape_default, getitem_1);  reshape_default = getitem_1 = None
        mul_tensor_1: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, -1);  reshape_default_1 = None
        mul_tensor_2: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default);  mul_tensor_1 = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_2: "f32[16, 3, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_1);  mul_tensor_2 = _shape_param_1 = None
        return (constant_pad_nd_default, reshape_default_2)


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
