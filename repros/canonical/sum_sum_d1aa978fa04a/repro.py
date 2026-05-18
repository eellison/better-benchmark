"""
Standalone repro captured via capture_hook.
Label: efficientnet_b0_training
Pattern hash: d1aa978fa04a
Shape hash: 5c4dfa27
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
    def forward(self, getitem_293: "f32[4, 24, 56, 56]", getitem_308: "f32[4, 24, 56, 56]", convolution_9: "f32[4, 24, 56, 56]", unsqueeze_714: "f32[1, 24, 1, 1]", squeeze_16: "f32[24]", primals_44: "f32[24]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        add_tensor: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_293, getitem_308);  getitem_293 = getitem_308 = None
        sum_dim_int_list: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 2, 3])
        sub_tensor: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_714);  convolution_9 = unsqueeze_714 = None
        mul_tensor: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_tensor, sub_tensor)
        sum_dim_int_list_1: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[24]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 7.971938775510203e-05);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[24]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 7.971938775510203e-05);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_tensor_4: "f32[24]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_44);  squeeze_16 = primals_44 = None
        unsqueeze_default_6: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_tensor, mul_tensor_6);  add_tensor = mul_tensor_6 = None
        sub_tensor_2: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7


def _default_make_inputs():
    return [
    torch.randn(301056, dtype=torch.float32, device='cuda').as_strided([4, 24, 56, 56], [75264, 1, 1344, 24]),  # getitem_293
    torch.randn(301056, dtype=torch.float32, device='cuda').as_strided([4, 24, 56, 56], [75264, 1, 1344, 24]),  # getitem_308
    torch.randn(301056, dtype=torch.float32, device='cuda').as_strided([4, 24, 56, 56], [75264, 1, 1344, 24]),  # convolution_9
    torch.randn([1, 24, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
