"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train
Pattern hash: 28d43d02650e
Shape hash: 03c36796
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 512, 12, 12], f32, stride=(73728, 1, 6144, 512)), T([128, 512, 24, 24], f32, stride=(294912, 1, 12288, 512)), T([128, 512, 24, 24], f32, stride=(294912, 1, 12288, 512)), T([128, 512, 24, 24], f32, stride=(294912, 1, 12288, 512)), T([128, 512, 1, 1], f32), T([], f32), T([128, 512, 24, 24], f32, stride=(294912, 1, 12288, 512)))"

class Repro(torch.nn.Module):
    def forward(self, getitem_282: "f32[128, 512, 12, 12]", mul_133: "f32[128, 512, 24, 24]", getitem_279: "f32[128, 512, 24, 24]", add_212: "f32[128, 512, 24, 24]", convolution_23: "f32[128, 512, 1, 1]", primals_70: "f32[]", convolution_21: "f32[128, 512, 24, 24]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:149 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_backward_default: "f32[128, 512, 24, 24]" = torch.ops.aten.avg_pool2d_backward.default(getitem_282, mul_133, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_282 = mul_133 = None
        add_tensor: "f32[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(getitem_279, avg_pool2d_backward_default);  getitem_279 = avg_pool2d_backward_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_tensor: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_tensor, 0.9622504486493761);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_tensor_1: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.7015043497085571);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_tensor_2: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_212);  mul_tensor_1 = add_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_tensor_3: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.2);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_default: "f32[128, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_tensor_4: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_3, primals_70);  mul_tensor_3 = primals_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_5: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 2.0);  mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor_6: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_5, convolution_21);  mul_tensor_5 = convolution_21 = None
        sum_dim_int_list: "f32[128, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [2, 3], True);  mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_tensor: "f32[128, 512, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_default)
        mul_tensor_7: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor);  sigmoid_default = sub_tensor = None
        mul_tensor_8: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_7);  sum_dim_int_list = mul_tensor_7 = None
        return mul_tensor_8


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
