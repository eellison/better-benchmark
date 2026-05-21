"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_unet_train_001
Pattern hash: 2c925f59efff
Shape hash: ee8575b6
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
_shapes_config = "(T([8, 1024, 80, 119], f32), T([80, 1], f32), T([118], f32), T([80, 1], i64, gen=Index(8)), T([118], i64, gen=Index(8)), T([118], i64, gen=Index(8)), T([80, 1], i64, gen=Index(8)), T([8, 512, 40, 59], f32), T([1, 512, 1, 1], f32), T([1, 512, 1, 1], f32), T([512], f32), T([512], f32), T([], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_24: "f32[8, 1024, 80, 119]", arg89_1: "f32[80, 1]", arg88_1: "f32[118]", arg85_1: "i64[80, 1]", arg87_1: "i64[118]", arg86_1: "i64[118]", arg84_1: "i64[80, 1]", arg81_1: "f32[8, 512, 40, 59]", arg82_1: "f32[1, 512, 1, 1]", arg83_1: "f32[1, 512, 1, 1]", arg24_1: "f32[512]", arg25_1: "f32[512]", full: "f32[]"):
        # No stacktrace found for following nodes
        slice_tensor: "f32[8, 512, 80, 119]" = torch.ops.aten.slice.Tensor(getitem_24, 1, 512, 1024);  getitem_24 = None
        constant_pad_nd_default: "f32[8, 512, 80, 118]" = torch.ops.aten.constant_pad_nd.default(slice_tensor, [0, -1, 0, 0]);  slice_tensor = None
        mul_tensor: "f32[8, 512, 80, 118]" = torch.ops.aten.mul.Tensor(constant_pad_nd_default, arg89_1);  arg89_1 = None
        neg_default: "f32[8, 512, 80, 118]" = torch.ops.aten.neg.default(mul_tensor)
        add_tensor: "f32[8, 512, 80, 118]" = torch.ops.aten.add.Tensor(constant_pad_nd_default, neg_default);  constant_pad_nd_default = neg_default = None
        mul_tensor_1: "f32[8, 512, 80, 118]" = torch.ops.aten.mul.Tensor(mul_tensor, arg88_1)
        neg_default_1: "f32[8, 512, 80, 118]" = torch.ops.aten.neg.default(mul_tensor_1)
        add_tensor_1: "f32[8, 512, 80, 118]" = torch.ops.aten.add.Tensor(mul_tensor, neg_default_1);  mul_tensor = neg_default_1 = None
        mul_tensor_2: "f32[8, 512, 80, 118]" = torch.ops.aten.mul.Tensor(add_tensor, arg88_1);  arg88_1 = None
        neg_default_2: "f32[8, 512, 80, 118]" = torch.ops.aten.neg.default(mul_tensor_2)
        add_tensor_2: "f32[8, 512, 80, 118]" = torch.ops.aten.add.Tensor(add_tensor, neg_default_2);  add_tensor = neg_default_2 = None
        full_default: "f32[8, 512, 40, 59]" = torch.ops.aten.full.default([8, 512, 40, 59], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[8, 512, 40, 59]" = torch.ops.aten.index_put.default(full_default, [None, None, arg85_1, arg87_1], mul_tensor_1, True);  mul_tensor_1 = None
        index_put_default_1: "f32[8, 512, 40, 59]" = torch.ops.aten.index_put.default(full_default, [None, None, arg85_1, arg86_1], add_tensor_1, True);  arg85_1 = add_tensor_1 = None
        add_tensor_3: "f32[8, 512, 40, 59]" = torch.ops.aten.add.Tensor(index_put_default, index_put_default_1);  index_put_default = index_put_default_1 = None
        index_put_default_2: "f32[8, 512, 40, 59]" = torch.ops.aten.index_put.default(full_default, [None, None, arg84_1, arg87_1], mul_tensor_2, True);  arg87_1 = mul_tensor_2 = None
        add_tensor_4: "f32[8, 512, 40, 59]" = torch.ops.aten.add.Tensor(add_tensor_3, index_put_default_2);  add_tensor_3 = index_put_default_2 = None
        index_put_default_3: "f32[8, 512, 40, 59]" = torch.ops.aten.index_put.default(full_default, [None, None, arg84_1, arg86_1], add_tensor_2, True);  full_default = arg84_1 = arg86_1 = add_tensor_2 = None
        add_tensor_5: "f32[8, 512, 40, 59]" = torch.ops.aten.add.Tensor(add_tensor_4, index_put_default_3);  add_tensor_4 = index_put_default_3 = None
        sub_tensor: "f32[8, 512, 40, 59]" = torch.ops.aten.sub.Tensor(arg81_1, arg82_1)
        mul_tensor_3: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(sub_tensor, arg83_1);  sub_tensor = None
        unsqueeze_default: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1)
        unsqueeze_default_1: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_4: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(mul_tensor_3, unsqueeze_default_1);  mul_tensor_3 = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_default_3: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_6: "f32[8, 512, 40, 59]" = torch.ops.aten.add.Tensor(mul_tensor_4, unsqueeze_default_3);  mul_tensor_4 = unsqueeze_default_3 = None
        relu_default: "f32[8, 512, 40, 59]" = torch.ops.aten.relu.default(add_tensor_6);  add_tensor_6 = None
        le_scalar: "b8[8, 512, 40, 59]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[8, 512, 40, 59]" = torch.ops.aten.where.self(le_scalar, full, add_tensor_5);  le_scalar = full = add_tensor_5 = None
        squeeze_dims: "f32[512]" = torch.ops.aten.squeeze.dims(arg82_1, [0, 2, 3]);  arg82_1 = None
        unsqueeze_default_4: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[8, 512, 40, 59]" = torch.ops.aten.sub.Tensor(arg81_1, unsqueeze_default_6);  arg81_1 = unsqueeze_default_6 = None
        mul_tensor_5: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 2, 3]);  mul_tensor_5 = None
        mul_tensor_6: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 5.296610169491525e-05);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_8: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_7: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 5.296610169491525e-05)
        squeeze_dims_1: "f32[512]" = torch.ops.aten.squeeze.dims(arg83_1, [0, 2, 3]);  arg83_1 = None
        mul_tensor_8: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_9: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_7, mul_tensor_8);  mul_tensor_7 = mul_tensor_8 = None
        unsqueeze_default_10: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_11: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_10: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, arg24_1);  arg24_1 = None
        unsqueeze_default_13: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_10, 0);  mul_tensor_10 = None
        unsqueeze_default_14: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_11: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[8, 512, 40, 59]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_11);  where_self = mul_tensor_11 = None
        sub_tensor_3: "f32[8, 512, 40, 59]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_12: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        mul_tensor_13: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, squeeze_dims_1);  sum_dim_int_list_1 = squeeze_dims_1 = None
        sum_dim_int_list_2: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 2, 3]);  mul_tensor_12 = None
        return (mul_tensor_13, sum_dim_int_list_2)



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
