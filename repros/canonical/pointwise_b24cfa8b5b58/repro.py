"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_inference
Pattern hash: b24cfa8b5b58
Shape hash: b166685a
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution_79: "f32[32, 1536, 1, 1]", convolution_77: "f32[32, 1536, 9, 9]", add_109: "f32[32, 1536, 9, 9]", view_168: "f32[1, 2304, 1536]", getitem_113: "f32[1, 2304, 1]", getitem_112: "f32[1, 2304, 1]", arg218_1: "f32[2304, 1, 1, 1]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_default: "f32[32, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_79);  convolution_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor: "f32[32, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_77, sigmoid_default);  convolution_77 = sigmoid_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_1: "f32[32, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_tensor_2: "f32[32, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 0.2);  mul_tensor_1 = None
        add_tensor: "f32[32, 1536, 9, 9]" = torch.ops.aten.add.Tensor(mul_tensor_2, add_109);  mul_tensor_2 = add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_tensor: "f32[1, 2304, 1536]" = torch.ops.aten.sub.Tensor(view_168, getitem_113);  view_168 = getitem_113 = None
        add_tensor_1: "f32[1, 2304, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
        rsqrt_default: "f32[1, 2304, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_3: "f32[1, 2304, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_4: "f32[2304, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg218_1, 0.04562504637317021);  arg218_1 = None
        reshape_default: "f32[2304]" = torch.ops.aten.reshape.default(mul_tensor_4, [-1]);  mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_default: "f32[2304, 1]" = torch.ops.aten.unsqueeze.default(reshape_default, -1);  reshape_default = None
        mul_tensor_5: "f32[1, 2304, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor_3, unsqueeze_default);  mul_tensor_3 = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        reshape_default_1: "f32[2304, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_5, _shape_param_0);  mul_tensor_5 = _shape_param_0 = None
        return (add_tensor, reshape_default_1)


def _default_make_inputs():
    return [
    torch.randn([32, 1536, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1536, 9, 9], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1536, 9, 9], dtype=torch.float32, device='cuda'),
    torch.randn([1, 2304, 1536], dtype=torch.float32, device='cuda'),
    torch.randn([1, 2304, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 2304, 1], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 1, 1, 1], dtype=torch.float32, device='cuda'),
    [2304, 1536, 1, 1],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
