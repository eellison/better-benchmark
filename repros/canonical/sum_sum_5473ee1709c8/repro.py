"""
Standalone repro captured via capture_hook.
Label: densenet121_training
Pattern hash: 5473ee1709c8
Shape hash: 3d652874
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
    def forward(self, mul_1592: "f32[4, 512, 28, 28]", mul_1610: "f32[4, 480, 28, 28]", mul_1628: "f32[4, 448, 28, 28]", mul_1646: "f32[4, 416, 28, 28]", mul_1664: "f32[4, 384, 28, 28]", mul_1682: "f32[4, 352, 28, 28]", mul_1700: "f32[4, 320, 28, 28]", mul_1718: "f32[4, 288, 28, 28]", relu_22: "f32[4, 256, 28, 28]", full_default: "f32[]", getitem_535: "f32[4, 256, 28, 28]", cat_9: "f32[4, 256, 28, 28]", unsqueeze_1662: "f32[1, 256, 1, 1]", squeeze_67: "f32[256]", primals_137: "f32[256]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        slice_tensor: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1592, 1, 224, 256);  mul_1592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_tensor_1: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1610, 1, 224, 256);  mul_1610 = None
        add_tensor: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None
        slice_tensor_2: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1628, 1, 224, 256);  mul_1628 = None
        add_tensor_1: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor, slice_tensor_2);  add_tensor = slice_tensor_2 = None
        slice_tensor_3: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1646, 1, 224, 256);  mul_1646 = None
        add_tensor_2: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_1, slice_tensor_3);  add_tensor_1 = slice_tensor_3 = None
        slice_tensor_4: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1664, 1, 224, 256);  mul_1664 = None
        add_tensor_3: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_2, slice_tensor_4);  add_tensor_2 = slice_tensor_4 = None
        slice_tensor_5: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1682, 1, 224, 256);  mul_1682 = None
        add_tensor_4: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_3, slice_tensor_5);  add_tensor_3 = slice_tensor_5 = None
        slice_tensor_6: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1700, 1, 224, 256);  mul_1700 = None
        add_tensor_5: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_4, slice_tensor_6);  add_tensor_4 = slice_tensor_6 = None
        slice_tensor_7: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1718, 1, 224, 256);  mul_1718 = None
        add_tensor_6: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_5, slice_tensor_7);  add_tensor_5 = slice_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        le_scalar: "b8[4, 256, 28, 28]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        where_self: "f32[4, 256, 28, 28]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_535);  le_scalar = full_default = getitem_535 = None
        sum_dim_int_list: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[4, 256, 28, 28]" = torch.ops.aten.sub.Tensor(cat_9, unsqueeze_1662);  cat_9 = unsqueeze_1662 = None
        mul_tensor: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00031887755102040814);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00031887755102040814);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_tensor_4: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_137);  squeeze_67 = primals_137 = None
        unsqueeze_default_6: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[4, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[4, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_tensor_8: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_tensor_7, 1, 224, 256);  mul_tensor_7 = None
        add_tensor_7: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_6, slice_tensor_8);  add_tensor_6 = slice_tensor_8 = None
        return add_tensor_7


def _default_make_inputs():
    return [
    torch.randn([4, 512, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 480, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 448, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 416, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 384, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 352, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 320, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 288, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 256, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([4, 256, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 256, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
