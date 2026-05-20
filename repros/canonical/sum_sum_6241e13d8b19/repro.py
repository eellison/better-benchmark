"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train
Pattern hash: 6241e13d8b19
Shape hash: c2977fbf
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([512, 16, 112, 112], f32, stride=(200704, 1, 1792, 16)), T([512, 16, 112, 112], f32, stride=(200704, 1, 1792, 16)), T([512, 16, 112, 112], f32, stride=(200704, 1, 1792, 16)), T([1, 16, 1, 1], f32), T([1, 16, 1, 1], f32), T([16], f32), T([16], f32), T([], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_269: "f32[512, 16, 112, 112]", getitem_275: "f32[512, 16, 112, 112]", convolution: "f32[512, 16, 112, 112]", getitem_1: "f32[1, 16, 1, 1]", rsqrt: "f32[1, 16, 1, 1]", primals_6: "f32[16]", primals_7: "f32[16]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:223 in forward, code: x = self.conv_dw(x)
        add_tensor: "f32[512, 16, 112, 112]" = torch.ops.aten.add.Tensor(getitem_269, getitem_275);  getitem_269 = getitem_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[512, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul_tensor: "f32[512, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        unsqueeze_default: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_default_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_default_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar: "b8[512, 16, 112, 112]" = torch.ops.aten.le.Scalar(add_tensor_1, -3)
        lt_scalar: "b8[512, 16, 112, 112]" = torch.ops.aten.lt.Scalar(add_tensor_1, 3)
        div_tensor: "f32[512, 16, 112, 112]" = torch.ops.aten.div.Tensor(add_tensor_1, 3);  add_tensor_1 = None
        add_tensor_2: "f32[512, 16, 112, 112]" = torch.ops.aten.add.Tensor(div_tensor, 0.5);  div_tensor = None
        mul_tensor_2: "f32[512, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_tensor, add_tensor_2);  add_tensor_2 = None
        where_self: "f32[512, 16, 112, 112]" = torch.ops.aten.where.self(lt_scalar, mul_tensor_2, add_tensor);  lt_scalar = mul_tensor_2 = add_tensor = None
        where_self_1: "f32[512, 16, 112, 112]" = torch.ops.aten.where.self(le_scalar, full_default, where_self);  le_scalar = full_default = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        unsqueeze_default_4: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_1: "f32[512, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_default_6);  convolution = unsqueeze_default_6 = None
        mul_tensor_3: "f32[512, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_1)
        sum_dim_int_list_1: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 2, 3]);  mul_tensor_3 = None
        mul_tensor_4: "f32[16]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 1.5570192920918366e-07);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_8: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_5: "f32[16]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 1.5570192920918366e-07);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_tensor_6: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_7: "f32[16]" = torch.ops.aten.mul.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None
        unsqueeze_default_10: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_11: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_8: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_6);  squeeze_dims_1 = primals_6 = None
        unsqueeze_default_13: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_8, 0);  mul_tensor_8 = None
        unsqueeze_default_14: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_9: "f32[512, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[512, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_9);  where_self_1 = mul_tensor_9 = None
        sub_tensor_3: "f32[512, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_10: "f32[512, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        return mul_tensor_10


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
