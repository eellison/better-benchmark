"""
Standalone repro captured via capture_hook.
Label: torchbench_shufflenet_v2_x1_0_train
Pattern hash: e59cea1e602b
Shape hash: 897b9ee4
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, view_64: "f32[512, 116, 28, 28]", getitem_290: "f32[512, 58, 28, 28]", convolution_4: "f32[512, 58, 28, 28]", unsqueeze_840: "f32[1, 58, 1, 1]", squeeze_13: "f32[58]", primals_30: "f32[58]", convolution_2: "f32[512, 58, 28, 28]", getitem_7: "f32[1, 58, 1, 1]", rsqrt_2: "f32[1, 58, 1, 1]", primals_18: "f32[58]", primals_19: "f32[58]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:97 in forward, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        slice_tensor: "f32[512, 58, 28, 28]" = torch.ops.aten.slice.Tensor(view_64, 1, 0, 58);  view_64 = None
        sum_dim_int_list: "f32[58]" = torch.ops.aten.sum.dim_IntList(getitem_290, [0, 2, 3])
        sub_tensor: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_840);  convolution_4 = unsqueeze_840 = None
        mul_tensor: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_290, sub_tensor)
        sum_dim_int_list_1: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[58]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.4912308673469386e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[58]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.4912308673469386e-06);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_tensor_4: "f32[58]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_30);  squeeze_13 = primals_30 = None
        unsqueeze_default_6: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_290, mul_tensor_6);  getitem_290 = mul_tensor_6 = None
        sub_tensor_2: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        sub_tensor_3: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_7)
        mul_tensor_8: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_2);  sub_tensor_3 = None
        unsqueeze_default_9: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_default_10: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, -1);  unsqueeze_default_9 = None
        mul_tensor_9: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_8, unsqueeze_default_10);  mul_tensor_8 = unsqueeze_default_10 = None
        unsqueeze_default_11: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_default_12: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, -1);  unsqueeze_default_11 = None
        add_tensor: "f32[512, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_9, unsqueeze_default_12);  mul_tensor_9 = unsqueeze_default_12 = None
        relu_default: "f32[512, 58, 28, 28]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        le_scalar: "b8[512, 58, 28, 28]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[512, 58, 28, 28]" = torch.ops.aten.where.self(le_scalar, full_default, slice_tensor);  le_scalar = full_default = slice_tensor = None
        squeeze_dims: "f32[58]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        unsqueeze_default_13: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_14: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        sum_dim_int_list_2: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_4: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_default_15);  convolution_2 = unsqueeze_default_15 = None
        mul_tensor_10: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_4)
        sum_dim_int_list_3: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 2, 3]);  mul_tensor_10 = None
        mul_tensor_11: "f32[58]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 2.4912308673469386e-06);  sum_dim_int_list_2 = None
        unsqueeze_default_16: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_tensor_11, 0);  mul_tensor_11 = None
        unsqueeze_default_17: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 3);  unsqueeze_default_17 = None
        mul_tensor_12: "f32[58]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 2.4912308673469386e-06);  sum_dim_int_list_3 = None
        squeeze_dims_1: "f32[58]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_tensor_13: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_14: "f32[58]" = torch.ops.aten.mul.Tensor(mul_tensor_12, mul_tensor_13);  mul_tensor_12 = mul_tensor_13 = None
        unsqueeze_default_19: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_tensor_14, 0);  mul_tensor_14 = None
        unsqueeze_default_20: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 2);  unsqueeze_default_19 = None
        unsqueeze_default_21: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 3);  unsqueeze_default_20 = None
        mul_tensor_15: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_18);  squeeze_dims_1 = primals_18 = None
        unsqueeze_default_22: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_tensor_15, 0);  mul_tensor_15 = None
        unsqueeze_default_23: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 3);  unsqueeze_default_23 = None
        mul_tensor_16: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_21);  sub_tensor_4 = unsqueeze_default_21 = None
        sub_tensor_5: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_16);  where_self = mul_tensor_16 = None
        sub_tensor_6: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_5, unsqueeze_default_18);  sub_tensor_5 = unsqueeze_default_18 = None
        mul_tensor_17: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_6, unsqueeze_default_24);  sub_tensor_6 = unsqueeze_default_24 = None
        return (mul_tensor_7, mul_tensor_17)


def _default_make_inputs():
    return [
    torch.randn([512, 116, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn(23281664, dtype=torch.float32, device='cuda').as_strided([512, 58, 28, 28], [45472, 1, 1624, 58]),  # getitem_290
    torch.randn(23281664, dtype=torch.float32, device='cuda').as_strided([512, 58, 28, 28], [45472, 1, 1624, 58]),  # convolution_4
    torch.randn([1, 58, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn(23281664, dtype=torch.float32, device='cuda').as_strided([512, 58, 28, 28], [45472, 1, 1624, 58]),  # convolution_2
    torch.randn([1, 58, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 58, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
