"""
Standalone repro captured via capture_hook.
Label: torchbench_densenet121_train
Pattern hash: 3dbde9d4687e
Shape hash: 93f2038d
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
_shapes_config = "(T([64, 512, 28, 28], f32), T([64, 480, 28, 28], f32), T([64, 448, 28, 28], f32), T([64, 416, 28, 28], f32), T([64, 384, 28, 28], f32), T([64, 352, 28, 28], f32), T([64, 320, 28, 28], f32), T([], f32), T([64, 320, 28, 28], f32), T([64, 320, 28, 28], f32), T([1, 320, 1, 1], f32), T([320], f32), T([320], f32))"

class Repro(torch.nn.Module):
    def forward(self, mul_1592: "f32[64, 512, 28, 28]", mul_1610: "f32[64, 480, 28, 28]", mul_1628: "f32[64, 448, 28, 28]", mul_1646: "f32[64, 416, 28, 28]", mul_1664: "f32[64, 384, 28, 28]", mul_1682: "f32[64, 352, 28, 28]", relu_26: "f32[64, 320, 28, 28]", full_default: "f32[]", getitem_523: "f32[64, 320, 28, 28]", cat_11: "f32[64, 320, 28, 28]", unsqueeze_1614: "f32[1, 320, 1, 1]", squeeze_79: "f32[320]", primals_161: "f32[320]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        slice_tensor: "f32[64, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1592, 1, 288, 320);  mul_1592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_tensor_1: "f32[64, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1610, 1, 288, 320);  mul_1610 = None
        add_tensor: "f32[64, 32, 28, 28]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None
        slice_tensor_2: "f32[64, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1628, 1, 288, 320);  mul_1628 = None
        add_tensor_1: "f32[64, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor, slice_tensor_2);  add_tensor = slice_tensor_2 = None
        slice_tensor_3: "f32[64, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1646, 1, 288, 320);  mul_1646 = None
        add_tensor_2: "f32[64, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_1, slice_tensor_3);  add_tensor_1 = slice_tensor_3 = None
        slice_tensor_4: "f32[64, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1664, 1, 288, 320);  mul_1664 = None
        add_tensor_3: "f32[64, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_2, slice_tensor_4);  add_tensor_2 = slice_tensor_4 = None
        slice_tensor_5: "f32[64, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1682, 1, 288, 320);  mul_1682 = None
        add_tensor_4: "f32[64, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_3, slice_tensor_5);  add_tensor_3 = slice_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        le_scalar: "b8[64, 320, 28, 28]" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
        where_self: "f32[64, 320, 28, 28]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_523);  le_scalar = full_default = getitem_523 = None
        sum_dim_int_list: "f32[320]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[64, 320, 28, 28]" = torch.ops.aten.sub.Tensor(cat_11, unsqueeze_1614);  cat_11 = unsqueeze_1614 = None
        mul_tensor: "f32[64, 320, 28, 28]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[320]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 1.992984693877551e-05);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[320]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 1.992984693877551e-05);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_tensor_4: "f32[320]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_161);  squeeze_79 = primals_161 = None
        unsqueeze_default_6: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[64, 320, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[64, 320, 28, 28]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[64, 320, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[64, 320, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_tensor_6: "f32[64, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_tensor_7, 1, 288, 320);  mul_tensor_7 = None
        add_tensor_5: "f32[64, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_4, slice_tensor_6);  add_tensor_4 = slice_tensor_6 = None
        return add_tensor_5



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
