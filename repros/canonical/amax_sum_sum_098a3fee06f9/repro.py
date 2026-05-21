"""
Standalone repro captured via capture_hook.
Label: hf_GPT2ForSequenceClassification_train_008
Pattern hash: 098a3fee06f9
Shape hash: b014e010
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
_shapes_config = "(T([], f32), T([], f32), T([8, 2], b8), T([8, 1], b8), T([8, 2], f32), T([8, 2], f32), T([8], i64, gen=Index(8)), T([8], i64, gen=Index(8)), S([8192, 2]))"

class Repro(torch.nn.Module):
    def forward(self, arg293_1: "f32[]", arg230_1: "f32[]", arg232_1: "b8[8, 2]", arg231_1: "b8[8, 1]", arg229_1: "f32[8, 2]", arg294_1: "f32[8, 2]", arg78_1: "i64[8]", arg228_1: "i64[8]", _shape_param_0):
        # No stacktrace found for following nodes
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(arg293_1, arg230_1);  arg293_1 = arg230_1 = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_self: "f32[8, 2]" = torch.ops.aten.where.self(arg232_1, scalar_tensor_default_1, scalar_tensor_default);  arg232_1 = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[8, 1]" = torch.ops.aten.where.self(arg231_1, div_tensor, full_default);  arg231_1 = div_tensor = full_default = None
        mul_tensor: "f32[8, 2]" = torch.ops.aten.mul.Tensor(where_self, where_self_1);  where_self = where_self_1 = None
        amax_default: "f32[8, 1]" = torch.ops.aten.amax.default(arg229_1, [1], True)
        sub_tensor: "f32[8, 2]" = torch.ops.aten.sub.Tensor(arg229_1, amax_default);  arg229_1 = amax_default = None
        exp_default: "f32[8, 2]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[8, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[8, 2]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        exp_default_1: "f32[8, 2]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list_1: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
        mul_tensor_1: "f32[8, 2]" = torch.ops.aten.mul.Tensor(exp_default_1, sum_dim_int_list_1);  exp_default_1 = sum_dim_int_list_1 = None
        sub_tensor_2: "f32[8, 2]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        add_tensor: "f32[8, 2]" = torch.ops.aten.add.Tensor(arg294_1, sub_tensor_2);  arg294_1 = sub_tensor_2 = None
        full_default_1: "f32[8, 1024, 2]" = torch.ops.aten.full.default([8, 1024, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[8, 1024, 2]" = torch.ops.aten.index_put.default(full_default_1, [arg78_1, arg228_1], add_tensor, True);  full_default_1 = arg78_1 = arg228_1 = add_tensor = None
        view_default: "f32[8192, 2]" = torch.ops.aten.view.default(index_put_default, _shape_param_0);  index_put_default = _shape_param_0 = None
        permute_default: "f32[2, 8192]" = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        return permute_default



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
