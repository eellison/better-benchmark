"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_training
Pattern hash: ee1a975a65fc
Shape hash: 4c7f0a82
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, add_471: "f32[32, 24, 56, 56]", convolution_11: "f32[32, 24, 56, 56]", unsqueeze_1138: "f32[1, 24, 1, 1]", squeeze_34: "f32[24]", primals_72: "f32[24]", getitem_421: "f32[32, 48, 112, 112]", convolution_6: "f32[32, 24, 112, 112]", getitem_13: "f32[1, 24, 1, 1]", rsqrt_6: "f32[1, 24, 1, 1]", primals_42: "f32[24]", primals_43: "f32[24]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        sum_dim_int_list: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_471, [0, 2, 3])
        sub_tensor: "f32[32, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1138);  convolution_11 = unsqueeze_1138 = None
        mul_tensor: "f32[32, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_471, sub_tensor)
        sum_dim_int_list_1: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[24]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 9.964923469387754e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[24]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 9.964923469387754e-06);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_tensor_4: "f32[24]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_72);  squeeze_34 = primals_72 = None
        unsqueeze_default_6: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[32, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[32, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_471, mul_tensor_6);  add_471 = mul_tensor_6 = None
        sub_tensor_2: "f32[32, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[32, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_tensor: "f32[32, 24, 112, 112]" = torch.ops.aten.slice.Tensor(getitem_421, 1, 24, 48);  getitem_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sub_tensor_3: "f32[32, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
        mul_tensor_8: "f32[32, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_6);  sub_tensor_3 = None
        unsqueeze_default_9: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1)
        unsqueeze_default_10: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, -1);  unsqueeze_default_9 = None
        mul_tensor_9: "f32[32, 24, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor_8, unsqueeze_default_10);  mul_tensor_8 = unsqueeze_default_10 = None
        unsqueeze_default_11: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
        unsqueeze_default_12: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, -1);  unsqueeze_default_11 = None
        add_tensor: "f32[32, 24, 112, 112]" = torch.ops.aten.add.Tensor(mul_tensor_9, unsqueeze_default_12);  mul_tensor_9 = unsqueeze_default_12 = None
        relu_default: "f32[32, 24, 112, 112]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        le_scalar: "b8[32, 24, 112, 112]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[32, 24, 112, 112]" = torch.ops.aten.where.self(le_scalar, full_default, slice_tensor);  le_scalar = full_default = slice_tensor = None
        squeeze_dims: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        unsqueeze_default_13: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_14: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        sum_dim_int_list_2: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_4: "f32[32, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_default_15);  convolution_6 = unsqueeze_default_15 = None
        mul_tensor_10: "f32[32, 24, 112, 112]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_4)
        sum_dim_int_list_3: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 2, 3]);  mul_tensor_10 = None
        mul_tensor_11: "f32[24]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 2.4912308673469386e-06);  sum_dim_int_list_2 = None
        unsqueeze_default_16: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_tensor_11, 0);  mul_tensor_11 = None
        unsqueeze_default_17: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 3);  unsqueeze_default_17 = None
        mul_tensor_12: "f32[24]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 2.4912308673469386e-06);  sum_dim_int_list_3 = None
        squeeze_dims_1: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_tensor_13: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_14: "f32[24]" = torch.ops.aten.mul.Tensor(mul_tensor_12, mul_tensor_13);  mul_tensor_12 = mul_tensor_13 = None
        unsqueeze_default_19: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_tensor_14, 0);  mul_tensor_14 = None
        unsqueeze_default_20: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 2);  unsqueeze_default_19 = None
        unsqueeze_default_21: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 3);  unsqueeze_default_20 = None
        mul_tensor_15: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_42);  squeeze_dims_1 = primals_42 = None
        unsqueeze_default_22: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_tensor_15, 0);  mul_tensor_15 = None
        unsqueeze_default_23: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 3);  unsqueeze_default_23 = None
        mul_tensor_16: "f32[32, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_21);  sub_tensor_4 = unsqueeze_default_21 = None
        sub_tensor_5: "f32[32, 24, 112, 112]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_16);  where_self = mul_tensor_16 = None
        sub_tensor_6: "f32[32, 24, 112, 112]" = torch.ops.aten.sub.Tensor(sub_tensor_5, unsqueeze_default_18);  sub_tensor_5 = unsqueeze_default_18 = None
        mul_tensor_17: "f32[32, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_6, unsqueeze_default_24);  sub_tensor_6 = unsqueeze_default_24 = None
        return (mul_tensor_7, mul_tensor_17)


def _default_make_inputs():
    return [
    torch.randn(2408448, dtype=torch.float32, device='cuda').as_strided([32, 24, 56, 56], [75264, 1, 1344, 24]),  # add_471
    torch.randn(2408448, dtype=torch.float32, device='cuda').as_strided([32, 24, 56, 56], [75264, 1, 1344, 24]),  # convolution_11
    torch.randn([1, 24, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn(19267584, dtype=torch.float32, device='cuda').as_strided([32, 48, 112, 112], [602112, 1, 5376, 48]),  # getitem_421
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([32, 24, 112, 112], [301056, 1, 2688, 24]),  # convolution_6
    torch.randn([1, 24, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 24, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
