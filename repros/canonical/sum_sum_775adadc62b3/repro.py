"""
Standalone repro captured via capture_hook.
Label: resnet18_training
Pattern hash: 775adadc62b3
Shape hash: c03d1933
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, where_12: "f32[4, 64, 56, 56]", getitem_90: "f32[4, 64, 56, 56]", relu_2: "f32[4, 64, 56, 56]", full_default: "f32[]", convolution_2: "f32[4, 64, 56, 56]", unsqueeze_286: "f32[1, 64, 1, 1]", squeeze_7: "f32[64]", primals_18: "f32[64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        add_tensor: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(where_12, getitem_90);  where_12 = getitem_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        le_scalar: "b8[4, 64, 56, 56]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_self: "f32[4, 64, 56, 56]" = torch.ops.aten.where.self(le_scalar, full_default, add_tensor);  le_scalar = full_default = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        sum_dim_int_list: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_286);  convolution_2 = unsqueeze_286 = None
        mul_tensor: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 7.971938775510203e-05);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 7.971938775510203e-05);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_tensor_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_18);  squeeze_7 = primals_18 = None
        unsqueeze_default_6: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7


def _default_make_inputs():
    return [
    torch.randn(802816, dtype=torch.float32, device='cuda').as_strided([4, 64, 56, 56], [200704, 1, 3584, 64]),  # where_12
    torch.randn(802816, dtype=torch.float32, device='cuda').as_strided([4, 64, 56, 56], [200704, 1, 3584, 64]),  # getitem_90
    torch.randn(802816, dtype=torch.float32, device='cuda').as_strided([4, 64, 56, 56], [200704, 1, 3584, 64]),  # relu_2
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(802816, dtype=torch.float32, device='cuda').as_strided([4, 64, 56, 56], [200704, 1, 3584, 64]),  # convolution_2
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
