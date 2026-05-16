"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_training
Pattern hash: de14911909b6
Shape hash: 2b1a28c2
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_321: "f32[32, 256, 28, 28]", mul_31: "f32[32, 256, 56, 56]", getitem_318: "f32[32, 256, 56, 56]", convolution_10: "f32[32, 256, 1, 1]", convolution_8: "f32[32, 256, 56, 56]", convolution_4: "f32[32, 256, 56, 56]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:149 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_backward_default: "f32[32, 256, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(getitem_321, mul_31, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_321 = mul_31 = None
        add_tensor: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(getitem_318, avg_pool2d_backward_default);  getitem_318 = avg_pool2d_backward_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_tensor: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(add_tensor, 0.9805806756909201);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_default: "f32[32, 256, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor_1: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_8, sigmoid_default)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_2: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 2.0);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_tensor_3: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.2);  mul_tensor_2 = None
        add_tensor_1: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor_3, convolution_4);  mul_tensor_3 = convolution_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_default: "f32[32, 256, 56, 56]" = torch.ops.aten.neg.default(add_tensor_1)
        exp_default: "f32[32, 256, 56, 56]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_2: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[32, 256, 56, 56]" = torch.ops.aten.reciprocal.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_4: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_5: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_4);  mul_tensor = None
        sub_tensor: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(1, mul_tensor_4);  mul_tensor_4 = None
        mul_tensor_6: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(add_tensor_1, sub_tensor);  add_tensor_1 = sub_tensor = None
        add_tensor_3: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor_6, 1);  mul_tensor_6 = None
        mul_tensor_7: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor_5, add_tensor_3);  mul_tensor_5 = add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_tensor_8: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.2);  mul_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_9: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor_8, 2.0);  mul_tensor_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor_10: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor_9, convolution_8);  mul_tensor_9 = convolution_8 = None
        sum_dim_int_list: "f32[32, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [2, 3], True);  mul_tensor_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_tensor_1: "f32[32, 256, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_default)
        mul_tensor_11: "f32[32, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor_1);  sigmoid_default = sub_tensor_1 = None
        mul_tensor_12: "f32[32, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_11);  sum_dim_int_list = mul_tensor_11 = None
        return mul_tensor_12


def _default_make_inputs():
    return [
    torch.randn([32, 256, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 56, 56], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
