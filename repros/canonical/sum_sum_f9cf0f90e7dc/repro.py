"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train
Pattern hash: f9cf0f90e7dc
Shape hash: 45c06ccb
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
_shapes_config = "(T([128, 256, 35, 35], f32, stride=(313600, 1, 8960, 256)), T([128, 256, 35, 35], f32, stride=(313600, 1, 8960, 256)), T([128, 256, 35, 35], f32, stride=(313600, 1, 8960, 256)), T([128, 256, 35, 35], f32, stride=(313600, 1, 8960, 256)), T([128, 256, 35, 35], f32, stride=(313600, 1, 8960, 256)), T([128, 96, 35, 35], f32, stride=(117600, 1, 3360, 96)), T([1, 96, 1, 1], f32), T([1, 96, 1, 1], f32), T([96], f32), T([96], f32), T([], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_421: "f32[128, 256, 35, 35]", cat: "f32[128, 256, 35, 35]", getitem_430: "f32[128, 256, 35, 35]", getitem_436: "f32[128, 256, 35, 35]", getitem_439: "f32[128, 256, 35, 35]", convolution_10: "f32[128, 96, 35, 35]", getitem_25: "f32[1, 96, 1, 1]", rsqrt_10: "f32[1, 96, 1, 1]", primals_66: "f32[96]", primals_67: "f32[96]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:57 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward_default: "f32[128, 256, 35, 35]" = torch.ops.aten.avg_pool2d_backward.default(getitem_421, cat, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_421 = cat = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        add_tensor: "f32[128, 256, 35, 35]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_default, getitem_430);  avg_pool2d_backward_default = getitem_430 = None
        add_tensor_1: "f32[128, 256, 35, 35]" = torch.ops.aten.add.Tensor(add_tensor, getitem_436);  add_tensor = getitem_436 = None
        add_tensor_2: "f32[128, 256, 35, 35]" = torch.ops.aten.add.Tensor(add_tensor_1, getitem_439);  add_tensor_1 = getitem_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:65 in forward, code: return torch.cat(outputs, 1)
        slice_tensor: "f32[128, 96, 35, 35]" = torch.ops.aten.slice.Tensor(add_tensor_2, 1, 128, 224);  add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[128, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_25)
        mul_tensor: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_10);  sub_tensor = None
        unsqueeze_default: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1)
        unsqueeze_default_1: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1);  primals_67 = None
        unsqueeze_default_3: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 96, 35, 35]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        le_scalar: "b8[128, 96, 35, 35]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[128, 96, 35, 35]" = torch.ops.aten.where.self(le_scalar, full_default, slice_tensor);  le_scalar = full_default = slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        unsqueeze_default_4: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[128, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_default_6);  convolution_10 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 6.3775510204081635e-06);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 6.3775510204081635e-06);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_tensor_5: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_66);  squeeze_dims_1 = primals_66 = None
        unsqueeze_default_13: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[128, 96, 35, 35]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[128, 96, 35, 35]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        return mul_tensor_9



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
