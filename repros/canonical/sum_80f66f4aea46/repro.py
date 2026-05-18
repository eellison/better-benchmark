"""
Standalone repro captured via capture_hook.
Label: efficientnet_b0_training
Pattern hash: 80f66f4aea46
Shape hash: 8e7d0631
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
    def forward(self, convolution_3: "f32[4, 32, 1, 1]", convolution_1: "f32[4, 32, 112, 112]", getitem_3: "f32[1, 32, 1, 1]", rsqrt_1: "f32[1, 32, 1, 1]", primals_12: "f32[32]", primals_13: "f32[32]", getitem_326: "f32[4, 32, 112, 112]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        sigmoid_default: "f32[4, 32, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_3);  convolution_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        sub_tensor: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3);  convolution_1 = getitem_3 = None
        mul_tensor: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_1);  sub_tensor = rsqrt_1 = None
        unsqueeze_default: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
        unsqueeze_default_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
        unsqueeze_default_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        neg_default: "f32[4, 32, 112, 112]" = torch.ops.aten.neg.default(add_tensor)
        exp_default: "f32[4, 32, 112, 112]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_1: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[4, 32, 112, 112]" = torch.ops.aten.div.Tensor(add_tensor, add_tensor_1);  add_tensor = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_tensor_2: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_326, div_tensor);  getitem_326 = div_tensor = None
        sum_dim_int_list: "f32[4, 32, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2, 3], True);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        sub_tensor_1: "f32[4, 32, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_default)
        mul_tensor_3: "f32[4, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor_1);  sigmoid_default = sub_tensor_1 = None
        mul_tensor_4: "f32[4, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_3);  sum_dim_int_list = mul_tensor_3 = None
        return mul_tensor_4


def _default_make_inputs():
    return [
    torch.randn([4, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(1605632, dtype=torch.float32, device='cuda').as_strided([4, 32, 112, 112], [401408, 1, 3584, 32]),  # convolution_1
    torch.randn([1, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32], dtype=torch.float32, device='cuda'),
    torch.randn([32], dtype=torch.float32, device='cuda'),
    torch.randn(1605632, dtype=torch.float32, device='cuda').as_strided([4, 32, 112, 112], [401408, 1, 3584, 32]),  # getitem_326
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
