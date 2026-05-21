"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_unet_train
Pattern hash: f414cb5f1a49
Shape hash: 8bfb0b5d
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
    def forward(self, getitem_56: "f32[8, 256, 320, 479]", clamp_max_11: "f32[320, 1]", clamp_max_10: "f32[478]", clamp_max_8: "i64[320, 1]", clamp_max_9: "i64[478]", convert_element_type_11: "i64[478]", convert_element_type_9: "i64[320, 1]", convolution_13: "f32[8, 128, 160, 239]", getitem_35: "f32[1, 128, 1, 1]", rsqrt_13: "f32[1, 128, 1, 1]", primals_98: "f32[128]", primals_99: "f32[128]", full_default: "f32[]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67 in forward, code: x = torch.cat([x2, x1], dim=1)
        slice_tensor: "f32[8, 128, 320, 479]" = torch.ops.aten.slice.Tensor(getitem_56, 1, 128, 256);  getitem_56 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[8, 128, 320, 478]" = torch.ops.aten.constant_pad_nd.default(slice_tensor, [0, -1, 0, 0]);  slice_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:58 in forward, code: x1 = self.up(x1)
        mul_tensor: "f32[8, 128, 320, 478]" = torch.ops.aten.mul.Tensor(constant_pad_nd_default, clamp_max_11);  clamp_max_11 = None
        neg_default: "f32[8, 128, 320, 478]" = torch.ops.aten.neg.default(mul_tensor)
        add_tensor: "f32[8, 128, 320, 478]" = torch.ops.aten.add.Tensor(constant_pad_nd_default, neg_default);  constant_pad_nd_default = neg_default = None
        mul_tensor_1: "f32[8, 128, 320, 478]" = torch.ops.aten.mul.Tensor(mul_tensor, clamp_max_10)
        neg_default_1: "f32[8, 128, 320, 478]" = torch.ops.aten.neg.default(mul_tensor_1)
        add_tensor_1: "f32[8, 128, 320, 478]" = torch.ops.aten.add.Tensor(mul_tensor, neg_default_1);  mul_tensor = neg_default_1 = None
        mul_tensor_2: "f32[8, 128, 320, 478]" = torch.ops.aten.mul.Tensor(add_tensor, clamp_max_10);  clamp_max_10 = None
        neg_default_2: "f32[8, 128, 320, 478]" = torch.ops.aten.neg.default(mul_tensor_2)
        add_tensor_2: "f32[8, 128, 320, 478]" = torch.ops.aten.add.Tensor(add_tensor, neg_default_2);  add_tensor = neg_default_2 = None
        full_default_1: "f32[8, 128, 160, 239]" = torch.ops.aten.full.default([8, 128, 160, 239], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[8, 128, 160, 239]" = torch.ops.aten.index_put.default(full_default_1, [None, None, clamp_max_8, clamp_max_9], mul_tensor_1, True);  mul_tensor_1 = None
        index_put_default_1: "f32[8, 128, 160, 239]" = torch.ops.aten.index_put.default(full_default_1, [None, None, clamp_max_8, convert_element_type_11], add_tensor_1, True);  clamp_max_8 = add_tensor_1 = None
        add_tensor_3: "f32[8, 128, 160, 239]" = torch.ops.aten.add.Tensor(index_put_default, index_put_default_1);  index_put_default = index_put_default_1 = None
        index_put_default_2: "f32[8, 128, 160, 239]" = torch.ops.aten.index_put.default(full_default_1, [None, None, convert_element_type_9, clamp_max_9], mul_tensor_2, True);  clamp_max_9 = mul_tensor_2 = None
        add_tensor_4: "f32[8, 128, 160, 239]" = torch.ops.aten.add.Tensor(add_tensor_3, index_put_default_2);  add_tensor_3 = index_put_default_2 = None
        index_put_default_3: "f32[8, 128, 160, 239]" = torch.ops.aten.index_put.default(full_default_1, [None, None, convert_element_type_9, convert_element_type_11], add_tensor_2, True);  full_default_1 = convert_element_type_9 = convert_element_type_11 = add_tensor_2 = None
        add_tensor_5: "f32[8, 128, 160, 239]" = torch.ops.aten.add.Tensor(add_tensor_4, index_put_default_3);  add_tensor_4 = index_put_default_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        sub_tensor: "f32[8, 128, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_35)
        mul_tensor_3: "f32[8, 128, 160, 239]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_13);  sub_tensor = None
        unsqueeze_default: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
        unsqueeze_default_1: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_4: "f32[8, 128, 160, 239]" = torch.ops.aten.mul.Tensor(mul_tensor_3, unsqueeze_default_1);  mul_tensor_3 = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
        unsqueeze_default_3: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_6: "f32[8, 128, 160, 239]" = torch.ops.aten.add.Tensor(mul_tensor_4, unsqueeze_default_3);  mul_tensor_4 = unsqueeze_default_3 = None
        relu_default: "f32[8, 128, 160, 239]" = torch.ops.aten.relu.default(add_tensor_6);  add_tensor_6 = None
        le_scalar: "b8[8, 128, 160, 239]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[8, 128, 160, 239]" = torch.ops.aten.where.self(le_scalar, full_default, add_tensor_5);  le_scalar = full_default = add_tensor_5 = None
        squeeze_dims: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        unsqueeze_default_4: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[8, 128, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_default_6);  convolution_13 = unsqueeze_default_6 = None
        mul_tensor_5: "f32[8, 128, 160, 239]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 2, 3]);  mul_tensor_5 = None
        mul_tensor_6: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 3.268828451882845e-06);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_8: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_7: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 3.268828451882845e-06);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_tensor_8: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_9: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_7, mul_tensor_8);  mul_tensor_7 = mul_tensor_8 = None
        unsqueeze_default_10: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_11: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_10: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_98);  squeeze_dims_1 = primals_98 = None
        unsqueeze_default_13: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_10, 0);  mul_tensor_10 = None
        unsqueeze_default_14: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_11: "f32[8, 128, 160, 239]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[8, 128, 160, 239]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_11);  where_self = mul_tensor_11 = None
        sub_tensor_3: "f32[8, 128, 160, 239]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_12: "f32[8, 128, 160, 239]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        return mul_tensor_12


def _default_make_inputs():
    return [
    torch.randn([8, 256, 320, 479], dtype=torch.float32, device='cuda'),
    torch.randn([320, 1], dtype=torch.float32, device='cuda'),
    torch.randn([478], dtype=torch.float32, device='cuda'),
    torch.randint(0, 8, [320, 1], dtype=torch.int64, device='cuda'),
    torch.randint(0, 8, [478], dtype=torch.int64, device='cuda'),
    torch.randint(0, 8, [478], dtype=torch.int64, device='cuda'),
    torch.randint(0, 8, [320, 1], dtype=torch.int64, device='cuda'),
    torch.randn([8, 128, 160, 239], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
