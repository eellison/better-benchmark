"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForSequenceClassification_train_000
Pattern hash: 5dfabd070411
Shape hash: 4184cd8b
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
_shapes_config = "(T([4096, 2], f32), T([32, 128], i64), T([32], i64, gen=Index(32)), T([32], i64, gen=Index(2)), T([], f32), S([32, 128, 2]), S([1, 2]), S([32, 2]))"

class Repro(torch.nn.Module):
    def forward(self, mm_72: "f32[4096, 2]", arg0_1: "i64[32, 128]", iota_1: "i64[32]", arg342_1: "i64[32]", full_1: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[32, 128, 2]" = torch.ops.aten.view.default(mm_72, _shape_param_0);  mm_72 = _shape_param_0 = None
        ne_scalar: "b8[32, 128]" = torch.ops.aten.ne.Scalar(arg0_1, 0);  arg0_1 = None
        convert_element_type_default: "i32[32, 128]" = torch.ops.prims.convert_element_type.default(ne_scalar, torch.int32);  ne_scalar = None
        iota_default: "i32[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        mul_tensor: "i32[32, 128]" = torch.ops.aten.mul.Tensor(iota_default, convert_element_type_default);  iota_default = convert_element_type_default = None
        argmax_default: "i64[32]" = torch.ops.aten.argmax.default(mul_tensor, -1);  mul_tensor = None
        index_tensor: "f32[32, 2]" = torch.ops.aten.index.Tensor(view_default, [iota_1, argmax_default]);  view_default = iota_1 = argmax_default = None
        amax_default: "f32[32, 1]" = torch.ops.aten.amax.default(index_tensor, [1], True)
        sub_tensor: "f32[32, 2]" = torch.ops.aten.sub.Tensor(index_tensor, amax_default);  index_tensor = amax_default = None
        exp_default: "f32[32, 2]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[32, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[32, 2]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar_1: "b8[32]" = torch.ops.aten.ne.Scalar(arg342_1, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[32]" = torch.ops.aten.where.self(ne_scalar_1, arg342_1, full_default)
        unsqueeze_default: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[32, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[32]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[32]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        where_self_1: "f32[32]" = torch.ops.aten.where.self(ne_scalar_1, neg_default, full_1);  neg_default = full_1 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar_1);  ne_scalar_1 = None
        convert_element_type_default_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default_1);  sum_default_1 = convert_element_type_default_1 = None
        unsqueeze_default_1: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(arg342_1, 1);  arg342_1 = None
        ne_scalar_2: "b8[32, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_default_1, -100)
        where_self_2: "i64[32, 1]" = torch.ops.aten.where.self(ne_scalar_2, unsqueeze_default_1, full_default);  ne_scalar_2 = unsqueeze_default_1 = full_default = None
        iota_default_1: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default_1: "i64[1, 2]" = torch.ops.aten.view.default(iota_default_1, _shape_param_1);  iota_default_1 = _shape_param_1 = None
        expand_default: "i64[32, 2]" = torch.ops.aten.expand.default(where_self_2, _shape_param_2);  where_self_2 = _shape_param_2 = None
        eq_tensor: "b8[32, 2]" = torch.ops.aten.eq.Tensor(expand_default, view_default_1);  expand_default = view_default_1 = None
        return (div_tensor, eq_tensor)



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
