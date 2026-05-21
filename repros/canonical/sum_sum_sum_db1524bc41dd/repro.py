"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_unet_train_001
Pattern hash: db1524bc41dd
Shape hash: 26df3fb3
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
_shapes_config = "(T([8, 64, 640, 959], f32), T([8, 64, 640, 959], f32), T([8, 64, 640, 959], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg135_1: "f32[8, 64, 640, 959]", getitem: "f32[8, 64, 640, 959]", arg133_1: "f32[8, 64, 640, 959]", arg136_1: "f32[1, 64, 1, 1]", arg134_1: "f32[64]", arg44_1: "f32[64]"):
        # No stacktrace found for following nodes
        le_scalar: "b8[8, 64, 640, 959]" = torch.ops.aten.le.Scalar(arg135_1, 0);  arg135_1 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 64, 640, 959]" = torch.ops.aten.where.self(le_scalar, full_default, getitem);  le_scalar = full_default = getitem = None
        sum_dim_int_list: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(arg133_1, arg136_1);  arg133_1 = arg136_1 = None
        mul_tensor: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.0366266944734097e-07);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.0366266944734097e-07)
        mul_tensor_3: "f32[64]" = torch.ops.aten.mul.Tensor(arg134_1, arg134_1)
        mul_tensor_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[64]" = torch.ops.aten.mul.Tensor(arg134_1, arg44_1);  arg44_1 = None
        unsqueeze_default_6: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        mul_tensor_8: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, arg134_1);  sum_dim_int_list_1 = arg134_1 = None
        sum_dim_int_list_2: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 2, 3]);  mul_tensor_7 = None
        return (mul_tensor_8, sum_dim_int_list_2)



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
