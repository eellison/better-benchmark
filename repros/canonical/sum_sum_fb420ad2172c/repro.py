"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_unet_train
Pattern hash: fb420ad2172c
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
    def forward(self, getitem_68: "f32[8, 1024, 80, 119]", clamp_max_3: "f32[80, 1]", clamp_max_2: "f32[118]", clamp_max: "i64[80, 1]", clamp_max_1: "i64[118]", convert_element_type_3: "i64[118]", convert_element_type_1: "i64[80, 1]", convolution_9: "f32[8, 512, 40, 59]", getitem_27: "f32[1, 512, 1, 1]", rsqrt_9: "f32[1, 512, 1, 1]", primals_70: "f32[512]", primals_71: "f32[512]", full_default: "f32[]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67 in forward, code: x = torch.cat([x2, x1], dim=1)
        slice_tensor: "f32[8, 512, 80, 119]" = torch.ops.aten.slice.Tensor(getitem_68, 1, 512, 1024);  getitem_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[8, 512, 80, 118]" = torch.ops.aten.constant_pad_nd.default(slice_tensor, [0, -1, 0, 0]);  slice_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:58 in forward, code: x1 = self.up(x1)
        mul_tensor: "f32[8, 512, 80, 118]" = torch.ops.aten.mul.Tensor(constant_pad_nd_default, clamp_max_3);  clamp_max_3 = None
        neg_default: "f32[8, 512, 80, 118]" = torch.ops.aten.neg.default(mul_tensor)
        add_tensor: "f32[8, 512, 80, 118]" = torch.ops.aten.add.Tensor(constant_pad_nd_default, neg_default);  constant_pad_nd_default = neg_default = None
        mul_tensor_1: "f32[8, 512, 80, 118]" = torch.ops.aten.mul.Tensor(mul_tensor, clamp_max_2)
        neg_default_1: "f32[8, 512, 80, 118]" = torch.ops.aten.neg.default(mul_tensor_1)
        add_tensor_1: "f32[8, 512, 80, 118]" = torch.ops.aten.add.Tensor(mul_tensor, neg_default_1);  mul_tensor = neg_default_1 = None
        mul_tensor_2: "f32[8, 512, 80, 118]" = torch.ops.aten.mul.Tensor(add_tensor, clamp_max_2);  clamp_max_2 = None
        neg_default_2: "f32[8, 512, 80, 118]" = torch.ops.aten.neg.default(mul_tensor_2)
        add_tensor_2: "f32[8, 512, 80, 118]" = torch.ops.aten.add.Tensor(add_tensor, neg_default_2);  add_tensor = neg_default_2 = None
        full_default_1: "f32[8, 512, 40, 59]" = torch.ops.aten.full.default([8, 512, 40, 59], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[8, 512, 40, 59]" = torch.ops.aten.index_put.default(full_default_1, [None, None, clamp_max, clamp_max_1], mul_tensor_1, True);  mul_tensor_1 = None
        index_put_default_1: "f32[8, 512, 40, 59]" = torch.ops.aten.index_put.default(full_default_1, [None, None, clamp_max, convert_element_type_3], add_tensor_1, True);  clamp_max = add_tensor_1 = None
        add_tensor_3: "f32[8, 512, 40, 59]" = torch.ops.aten.add.Tensor(index_put_default, index_put_default_1);  index_put_default = index_put_default_1 = None
        index_put_default_2: "f32[8, 512, 40, 59]" = torch.ops.aten.index_put.default(full_default_1, [None, None, convert_element_type_1, clamp_max_1], mul_tensor_2, True);  clamp_max_1 = mul_tensor_2 = None
        add_tensor_4: "f32[8, 512, 40, 59]" = torch.ops.aten.add.Tensor(add_tensor_3, index_put_default_2);  add_tensor_3 = index_put_default_2 = None
        index_put_default_3: "f32[8, 512, 40, 59]" = torch.ops.aten.index_put.default(full_default_1, [None, None, convert_element_type_1, convert_element_type_3], add_tensor_2, True);  full_default_1 = convert_element_type_1 = convert_element_type_3 = add_tensor_2 = None
        add_tensor_5: "f32[8, 512, 40, 59]" = torch.ops.aten.add.Tensor(add_tensor_4, index_put_default_3);  add_tensor_4 = index_put_default_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        sub_tensor: "f32[8, 512, 40, 59]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_27)
        mul_tensor_3: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_9);  sub_tensor = None
        unsqueeze_default: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1)
        unsqueeze_default_1: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_4: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(mul_tensor_3, unsqueeze_default_1);  mul_tensor_3 = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1);  primals_71 = None
        unsqueeze_default_3: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_6: "f32[8, 512, 40, 59]" = torch.ops.aten.add.Tensor(mul_tensor_4, unsqueeze_default_3);  mul_tensor_4 = unsqueeze_default_3 = None
        relu_default: "f32[8, 512, 40, 59]" = torch.ops.aten.relu.default(add_tensor_6);  add_tensor_6 = None
        le_scalar: "b8[8, 512, 40, 59]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[8, 512, 40, 59]" = torch.ops.aten.where.self(le_scalar, full_default, add_tensor_5);  le_scalar = full_default = add_tensor_5 = None
        squeeze_dims: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        unsqueeze_default_4: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[8, 512, 40, 59]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_default_6);  convolution_9 = unsqueeze_default_6 = None
        mul_tensor_5: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 2, 3]);  mul_tensor_5 = None
        mul_tensor_6: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 5.296610169491525e-05);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_8: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_7: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 5.296610169491525e-05);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_tensor_8: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_9: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_7, mul_tensor_8);  mul_tensor_7 = mul_tensor_8 = None
        unsqueeze_default_10: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_11: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_10: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_70);  squeeze_dims_1 = primals_70 = None
        unsqueeze_default_13: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_10, 0);  mul_tensor_10 = None
        unsqueeze_default_14: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_11: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[8, 512, 40, 59]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_11);  where_self = mul_tensor_11 = None
        sub_tensor_3: "f32[8, 512, 40, 59]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_12: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        return mul_tensor_12



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
