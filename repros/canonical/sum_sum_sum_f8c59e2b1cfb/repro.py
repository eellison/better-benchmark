"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_training
Pattern hash: f8c59e2b1cfb
Shape hash: ef84025e
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_196: "f32[32, 2048, 8, 8]", cat_11: "f32[32, 2048, 8, 8]", getitem_208: "f32[32, 2048, 8, 8]", getitem_217: "f32[32, 2048, 8, 8]", getitem_220: "f32[32, 2048, 8, 8]", convolution_83: "f32[32, 384, 8, 8]", getitem_175: "f32[1, 384, 1, 1]", rsqrt_83: "f32[1, 384, 1, 1]", primals_504: "f32[384]", primals_505: "f32[384]", full_default: "f32[]", convolution_82: "f32[32, 384, 8, 8]", getitem_173: "f32[1, 384, 1, 1]", rsqrt_82: "f32[1, 384, 1, 1]", primals_498: "f32[384]", primals_499: "f32[384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:236 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward_default: "f32[32, 2048, 8, 8]" = torch.ops.aten.avg_pool2d_backward.default(getitem_196, cat_11, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_196 = cat_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        add_tensor: "f32[32, 2048, 8, 8]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_default, getitem_208);  avg_pool2d_backward_default = getitem_208 = None
        add_tensor_1: "f32[32, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_tensor, getitem_217);  add_tensor = getitem_217 = None
        add_tensor_2: "f32[32, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_tensor_1, getitem_220);  add_tensor_1 = getitem_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:244 in forward, code: return torch.cat(outputs, 1)
        slice_tensor: "f32[32, 768, 8, 8]" = torch.ops.aten.slice.Tensor(add_tensor_2, 1, 1088, 1856);  add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:234 in _forward, code: branch3x3dbl = torch.cat(branch3x3dbl, 1)
        slice_tensor_1: "f32[32, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_tensor, 1, 0, 384)
        slice_tensor_2: "f32[32, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_tensor, 1, 384, 768);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_83, getitem_175)
        mul_tensor: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_83);  sub_tensor = None
        unsqueeze_default: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_504, -1)
        unsqueeze_default_1: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_505, -1);  primals_505 = None
        unsqueeze_default_3: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[32, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        le_scalar: "b8[32, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[32, 384, 8, 8]" = torch.ops.aten.where.self(le_scalar, full_default, slice_tensor_2);  le_scalar = slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_175, [0, 2, 3]);  getitem_175 = None
        unsqueeze_default_4: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_default_6);  convolution_83 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00048828125);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00048828125);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_83, [0, 2, 3]);  rsqrt_83 = None
        mul_tensor_5: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_504);  squeeze_dims_1 = primals_504 = None
        unsqueeze_default_13: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        sub_tensor_4: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_82, getitem_173)
        mul_tensor_10: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt_82);  sub_tensor_4 = None
        unsqueeze_default_16: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_498, -1)
        unsqueeze_default_17: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, -1);  unsqueeze_default_16 = None
        mul_tensor_11: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_10, unsqueeze_default_17);  mul_tensor_10 = unsqueeze_default_17 = None
        unsqueeze_default_18: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_499, -1);  primals_499 = None
        unsqueeze_default_19: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, -1);  unsqueeze_default_18 = None
        add_tensor_4: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_11, unsqueeze_default_19);  mul_tensor_11 = unsqueeze_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_1: "f32[32, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_4);  add_tensor_4 = None
        le_scalar_1: "b8[32, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_default_1, 0);  relu_default_1 = None
        where_self_1: "f32[32, 384, 8, 8]" = torch.ops.aten.where.self(le_scalar_1, full_default, slice_tensor_1);  le_scalar_1 = full_default = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_2: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_173, [0, 2, 3]);  getitem_173 = None
        unsqueeze_default_20: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_2, 0);  squeeze_dims_2 = None
        unsqueeze_default_21: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 2);  unsqueeze_default_20 = None
        unsqueeze_default_22: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_21, 3);  unsqueeze_default_21 = None
        sum_dim_int_list_2: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_5: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_default_22);  convolution_82 = unsqueeze_default_22 = None
        mul_tensor_12: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_5)
        sum_dim_int_list_3: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 2, 3]);  mul_tensor_12 = None
        mul_tensor_13: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 0.00048828125);  sum_dim_int_list_2 = None
        unsqueeze_default_23: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_24: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 2);  unsqueeze_default_23 = None
        unsqueeze_default_25: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, 3);  unsqueeze_default_24 = None
        mul_tensor_14: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 0.00048828125);  sum_dim_int_list_3 = None
        squeeze_dims_3: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_82, [0, 2, 3]);  rsqrt_82 = None
        mul_tensor_15: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, squeeze_dims_3)
        mul_tensor_16: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_14, mul_tensor_15);  mul_tensor_14 = mul_tensor_15 = None
        unsqueeze_default_26: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_16, 0);  mul_tensor_16 = None
        unsqueeze_default_27: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 2);  unsqueeze_default_26 = None
        unsqueeze_default_28: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_27, 3);  unsqueeze_default_27 = None
        mul_tensor_17: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, primals_498);  squeeze_dims_3 = primals_498 = None
        unsqueeze_default_29: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_17, 0);  mul_tensor_17 = None
        unsqueeze_default_30: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 2);  unsqueeze_default_29 = None
        unsqueeze_default_31: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, 3);  unsqueeze_default_30 = None
        mul_tensor_18: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_28);  sub_tensor_5 = unsqueeze_default_28 = None
        sub_tensor_6: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_18);  where_self_1 = mul_tensor_18 = None
        sub_tensor_7: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_6, unsqueeze_default_25);  sub_tensor_6 = unsqueeze_default_25 = None
        mul_tensor_19: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_7, unsqueeze_default_31);  sub_tensor_7 = unsqueeze_default_31 = None
        return (mul_tensor_9, mul_tensor_19)


def _default_make_inputs():
    return [
    torch.randn(4194304, dtype=torch.float32, device='cuda').as_strided([32, 2048, 8, 8], [131072, 1, 16384, 2048]),  # getitem_196
    torch.randn(4194304, dtype=torch.float32, device='cuda').as_strided([32, 2048, 8, 8], [131072, 1, 16384, 2048]),  # cat_11
    torch.randn(4194304, dtype=torch.float32, device='cuda').as_strided([32, 2048, 8, 8], [131072, 1, 16384, 2048]),  # getitem_208
    torch.randn(4194304, dtype=torch.float32, device='cuda').as_strided([32, 2048, 8, 8], [131072, 1, 16384, 2048]),  # getitem_217
    torch.randn(4194304, dtype=torch.float32, device='cuda').as_strided([32, 2048, 8, 8], [131072, 1, 16384, 2048]),  # getitem_220
    torch.randn(786432, dtype=torch.float32, device='cuda').as_strided([32, 384, 8, 8], [24576, 1, 3072, 384]),  # convolution_83
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(786432, dtype=torch.float32, device='cuda').as_strided([32, 384, 8, 8], [24576, 1, 3072, 384]),  # convolution_82
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
