"""
Standalone repro captured via capture_hook.
Label: resnet18_training
Pattern hash: c65e380ac58f
Shape hash: 21eafd01
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, where_10: "f32[4, 128, 28, 28]", convolution_7: "f32[4, 128, 28, 28]", unsqueeze_226: "f32[1, 128, 1, 1]", squeeze_22: "f32[128]", primals_48: "f32[128]", relu_5: "f32[4, 128, 28, 28]", full_default: "f32[]", getitem_81: "f32[4, 128, 28, 28]", convolution_5: "f32[4, 128, 28, 28]", unsqueeze_250: "f32[1, 128, 1, 1]", squeeze_16: "f32[128]", primals_36: "f32[128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        sum_dim_int_list: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        sub_tensor: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_226);  convolution_7 = unsqueeze_226 = None
        mul_tensor: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_10, sub_tensor)
        sum_dim_int_list_1: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00031887755102040814);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00031887755102040814);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_tensor_4: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_48);  squeeze_22 = primals_48 = None
        unsqueeze_default_6: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_10, mul_tensor_6);  where_10 = mul_tensor_6 = None
        sub_tensor_2: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        le_scalar: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_self: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_81);  le_scalar = full_default = getitem_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        sum_dim_int_list_2: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_3: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_250);  convolution_5 = unsqueeze_250 = None
        mul_tensor_8: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_3)
        sum_dim_int_list_3: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 2, 3]);  mul_tensor_8 = None
        mul_tensor_9: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 0.00031887755102040814);  sum_dim_int_list_2 = None
        unsqueeze_default_9: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_10: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_10: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 0.00031887755102040814);  sum_dim_int_list_3 = None
        mul_tensor_11: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_tensor_12: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        unsqueeze_default_12: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_12, 0);  mul_tensor_12 = None
        unsqueeze_default_13: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None
        mul_tensor_13: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_36);  squeeze_16 = primals_36 = None
        unsqueeze_default_15: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_16: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None
        mul_tensor_14: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_14);  sub_tensor_3 = unsqueeze_default_14 = None
        sub_tensor_4: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_14);  where_self = mul_tensor_14 = None
        sub_tensor_5: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_4, unsqueeze_default_11);  sub_tensor_4 = unsqueeze_default_11 = None
        mul_tensor_15: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_17);  sub_tensor_5 = unsqueeze_default_17 = None
        return (mul_tensor_7, mul_tensor_15)


def _default_make_inputs():
    return [
    torch.randn(401408, dtype=torch.float32, device='cuda').as_strided([4, 128, 28, 28], [100352, 1, 3584, 128]),  # where_10
    torch.randn(401408, dtype=torch.float32, device='cuda').as_strided([4, 128, 28, 28], [100352, 1, 3584, 128]),  # convolution_7
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn(401408, dtype=torch.float32, device='cuda').as_strided([4, 128, 28, 28], [100352, 1, 3584, 128]),  # relu_5
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(401408, dtype=torch.float32, device='cuda').as_strided([4, 128, 28, 28], [100352, 1, 3584, 128]),  # getitem_81
    torch.randn(401408, dtype=torch.float32, device='cuda').as_strided([4, 128, 28, 28], [100352, 1, 3584, 128]),  # convolution_5
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
