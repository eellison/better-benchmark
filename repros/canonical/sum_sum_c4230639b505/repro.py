"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_unet_train
Pattern hash: c4230639b505
Shape hash: 5aad63de
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
    def forward(self, getitem_62: "f32[8, 512, 160, 239]", getitem_80: "f32[8, 256, 80, 119]", getitem_17: "i8[8, 256, 80, 119]", convolution_5: "f32[8, 256, 160, 239]", getitem_15: "f32[1, 256, 1, 1]", rsqrt_5: "f32[1, 256, 1, 1]", primals_42: "f32[256]", primals_43: "f32[256]", full_default: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67 in forward, code: x = torch.cat([x2, x1], dim=1)
        slice_tensor: "f32[8, 256, 160, 239]" = torch.ops.aten.slice.Tensor(getitem_62, 1, 0, 256);  getitem_62 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:38 in forward, code: return self.maxpool_conv(x)
        full_default_1: "f32[2048, 38240]" = torch.ops.aten.full.default([2048, 38240], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        reshape_default: "f32[2048, 9520]" = torch.ops.aten.reshape.default(getitem_80, _shape_param_0);  getitem_80 = _shape_param_0 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[8, 256, 80, 119]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_17, [2, 2], [160, 239], [2, 2], [0, 0], [1, 1]);  getitem_17 = None
        reshape_default_1: "i64[2048, 9520]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_default, _shape_param_1);  _low_memory_max_pool_offsets_to_indices_default = _shape_param_1 = None
        scatter_add_default: "f32[2048, 38240]" = torch.ops.aten.scatter_add.default(full_default_1, 1, reshape_default_1, reshape_default);  full_default_1 = reshape_default_1 = reshape_default = None
        reshape_default_2: "f32[8, 256, 160, 239]" = torch.ops.aten.reshape.default(scatter_add_default, _shape_param_2);  scatter_add_default = _shape_param_2 = None
        add_tensor: "f32[8, 256, 160, 239]" = torch.ops.aten.add.Tensor(slice_tensor, reshape_default_2);  slice_tensor = reshape_default_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        sub_tensor: "f32[8, 256, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_15)
        mul_tensor: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_5);  sub_tensor = None
        unsqueeze_default: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1)
        unsqueeze_default_1: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
        unsqueeze_default_3: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[8, 256, 160, 239]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[8, 256, 160, 239]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        le_scalar: "b8[8, 256, 160, 239]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[8, 256, 160, 239]" = torch.ops.aten.where.self(le_scalar, full_default, add_tensor);  le_scalar = full_default = add_tensor = None
        squeeze_dims: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        unsqueeze_default_4: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[8, 256, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_default_6);  convolution_5 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 3.268828451882845e-06);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 3.268828451882845e-06);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_tensor_5: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_42);  squeeze_dims_1 = primals_42 = None
        unsqueeze_default_13: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[8, 256, 160, 239]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[8, 256, 160, 239]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        return mul_tensor_9


def _default_make_inputs():
    return [
    torch.randn([8, 512, 160, 239], dtype=torch.float32, device='cuda'),
    torch.randn([8, 256, 80, 119], dtype=torch.float32, device='cuda'),
    torch.randint(0, 4, (19660720,), dtype=torch.int8, device='cuda').as_strided([8, 256, 80, 119], [2457600, 9600, 119, 1]),  # getitem_17
    torch.randn([8, 256, 160, 239], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    [2048, 9520],  # _shape_param_0
    [2048, 9520],  # _shape_param_1
    [8, 256, 160, 239],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
