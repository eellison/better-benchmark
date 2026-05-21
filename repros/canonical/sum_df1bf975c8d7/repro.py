"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_train
Pattern hash: df1bf975c8d7
Shape hash: ea4a2f54
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 32, 112, 112], f32, stride=(401408, 1, 3584, 32)), T([1, 32, 1, 1], f32), T([1, 32, 1, 1], f32), T([32], f32), T([32], f32), T([128, 32, 112, 112], f32, stride=(401408, 1, 3584, 32)), T([128, 32, 1, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_1: "f32[128, 32, 112, 112]", getitem_3: "f32[1, 32, 1, 1]", rsqrt_1: "f32[1, 32, 1, 1]", primals_12: "f32[32]", primals_13: "f32[32]", getitem_326: "f32[128, 32, 112, 112]", convolution_3: "f32[128, 32, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3);  convolution_1 = getitem_3 = None
        mul_tensor: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_1);  sub_tensor = rsqrt_1 = None
        unsqueeze_default: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
        unsqueeze_default_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
        unsqueeze_default_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_default: "f32[128, 32, 112, 112]" = torch.ops.aten.neg.default(add_tensor)
        exp_default: "f32[128, 32, 112, 112]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_1: "f32[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 32, 112, 112]" = torch.ops.aten.div.Tensor(add_tensor, add_tensor_1);  add_tensor = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_tensor_2: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_326, div_tensor);  getitem_326 = div_tensor = None
        sigmoid_default: "f32[128, 32, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_3);  convolution_3 = None
        sum_dim_int_list: "f32[128, 32, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2, 3], True);  mul_tensor_2 = None
        sub_tensor_1: "f32[128, 32, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_default)
        mul_tensor_3: "f32[128, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor_1);  sigmoid_default = sub_tensor_1 = None
        mul_tensor_4: "f32[128, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_3);  sum_dim_int_list = mul_tensor_3 = None
        return mul_tensor_4



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
