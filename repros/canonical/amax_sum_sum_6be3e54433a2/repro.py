"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForSequenceClassification_train_001
Pattern hash: 6be3e54433a2
Shape hash: 05727731
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([], f32), T([], f32), T([32, 2], b8), T([32, 1], b8), T([32, 2], f32), T([32, 2], f32), T([32], i64, gen=Index(32)), T([32], i64, gen=Index(32)), S([4096, 2]))"

class Repro(torch.nn.Module):
    def forward(self, arg543_1: "f32[]", arg420_1: "f32[]", arg422_1: "b8[32, 2]", arg421_1: "b8[32, 1]", arg419_1: "f32[32, 2]", arg544_1: "f32[32, 2]", arg221_1: "i64[32]", arg418_1: "i64[32]", _shape_param_0):
        # No stacktrace found for following nodes
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(arg543_1, arg420_1);  arg543_1 = arg420_1 = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_self: "f32[32, 2]" = torch.ops.aten.where.self(arg422_1, scalar_tensor_default_1, scalar_tensor_default);  arg422_1 = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[32, 1]" = torch.ops.aten.where.self(arg421_1, div_tensor, full_default);  arg421_1 = div_tensor = full_default = None
        mul_tensor: "f32[32, 2]" = torch.ops.aten.mul.Tensor(where_self, where_self_1);  where_self = where_self_1 = None
        amax_default: "f32[32, 1]" = torch.ops.aten.amax.default(arg419_1, [1], True)
        sub_tensor: "f32[32, 2]" = torch.ops.aten.sub.Tensor(arg419_1, amax_default);  arg419_1 = amax_default = None
        exp_default: "f32[32, 2]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[32, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[32, 2]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        exp_default_1: "f32[32, 2]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list_1: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
        mul_tensor_1: "f32[32, 2]" = torch.ops.aten.mul.Tensor(exp_default_1, sum_dim_int_list_1);  exp_default_1 = sum_dim_int_list_1 = None
        sub_tensor_2: "f32[32, 2]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        add_tensor: "f32[32, 2]" = torch.ops.aten.add.Tensor(arg544_1, sub_tensor_2);  arg544_1 = sub_tensor_2 = None
        full_default_1: "f32[32, 128, 2]" = torch.ops.aten.full.default([32, 128, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32, 128, 2]" = torch.ops.aten.index_put.default(full_default_1, [arg221_1, arg418_1], add_tensor, True);  full_default_1 = arg221_1 = arg418_1 = add_tensor = None
        view_default: "f32[4096, 2]" = torch.ops.aten.view.default(index_put_default, _shape_param_0);  index_put_default = _shape_param_0 = None
        permute_default: "f32[2, 4096]" = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        constant_pad_nd_default: "f32[4, 4096]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 0, 0, 2]);  permute_default = None
        return constant_pad_nd_default

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
