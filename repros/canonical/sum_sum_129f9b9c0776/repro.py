"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_regnet_train
Pattern hash: 129f9b9c0776
Shape hash: 47feeedc
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
_shapes_config = "(T([1, 2240, 1, 1], f32), T([32, 2240, 7, 7], f32), T([32, 2240, 7, 7], f32), T([1, 2240, 1, 1], f32), T([2240], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_123: "f32[1, 2240, 1, 1]", where: "f32[32, 2240, 7, 7]", convolution_99: "f32[32, 2240, 7, 7]", rsqrt_61: "f32[1, 2240, 1, 1]", primals_448: "f32[2240]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[2240]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
        unsqueeze_default: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_1: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        sum_dim_int_list: "f32[2240]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        sub_tensor: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_default_2);  convolution_99 = unsqueeze_default_2 = None
        mul_tensor: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_tensor)
        sum_dim_int_list_1: "f32[2240]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.0006377551020408163);  sum_dim_int_list = None
        unsqueeze_default_3: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_4: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.0006377551020408163);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[2240]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
        mul_tensor_3: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_4: "f32[2240]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_6: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_7: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_5: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_448);  squeeze_dims_1 = primals_448 = None
        unsqueeze_default_9: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_6: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_8);  sub_tensor = unsqueeze_default_8 = None
        sub_tensor_1: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_tensor_6);  where = mul_tensor_6 = None
        sub_tensor_2: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_5);  sub_tensor_1 = unsqueeze_default_5 = None
        mul_tensor_7: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_11);  sub_tensor_2 = unsqueeze_default_11 = None
        return mul_tensor_7



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
