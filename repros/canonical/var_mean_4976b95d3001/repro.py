"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_infer
Pattern hash: 4976b95d3001
Shape hash: bf394a43
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([384, 1536, 1, 1], f32), T([128, 1536, 1, 1], f32), T([128, 1536, 7, 7], f32, stride=(75264, 1, 10752, 1536)), T([128, 1536, 7, 7], f32, stride=(75264, 1, 10752, 1536)), T([384, 1, 1, 1], f32), S([1, 384, -1]), S([384, 1536, 1, 1]))"

class Repro(torch.nn.Module):
    def forward(self, arg201_1: "f32[384, 1536, 1, 1]", convolution_73: "f32[128, 1536, 1, 1]", convolution_71: "f32[128, 1536, 7, 7]", add_100: "f32[128, 1536, 7, 7]", arg202_1: "f32[384, 1, 1, 1]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(arg201_1, _shape_param_0);  arg201_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [0, 2], correction = 0, keepdim = True)
        getitem: "f32[1, 384, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 384, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_default: "f32[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_71, sigmoid_default);  convolution_71 = sigmoid_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_1: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_tensor_2: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 0.2);  mul_tensor_1 = None
        add_tensor: "f32[128, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, add_100);  mul_tensor_2 = add_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_default: "f32[128, 1536, 7, 7]" = torch.ops.aten.neg.default(add_tensor)
        exp_default: "f32[128, 1536, 7, 7]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_1: "f32[128, 1536, 7, 7]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 1536, 7, 7]" = torch.ops.aten.div.Tensor(add_tensor, add_tensor_1);  add_tensor = add_tensor_1 = None
        mul_tensor_3: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(div_tensor, 0.9622504486493761);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_tensor: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(reshape_default, getitem_1);  reshape_default = getitem_1 = None
        add_tensor_2: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_4: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_5: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg202_1, 0.04562504637317021);  arg202_1 = None
        reshape_default_1: "f32[384]" = torch.ops.aten.reshape.default(mul_tensor_5, [-1]);  mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_default: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, -1);  reshape_default_1 = None
        mul_tensor_6: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default);  mul_tensor_4 = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        reshape_default_2: "f32[384, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_1);  mul_tensor_6 = _shape_param_1 = None
        return (mul_tensor_3, reshape_default_2)


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
