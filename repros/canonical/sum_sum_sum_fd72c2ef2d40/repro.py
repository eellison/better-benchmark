"""
Standalone repro captured via capture_hook.
Label: hf_GPT2ForSequenceClassification_infer_000
Pattern hash: fd72c2ef2d40
Shape hash: 03f102c9
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
_shapes_config = "(T([8], i64, gen=Index(2)), T([8192, 2], f32), T([8, 1024], i64), S([8, 1024, 2]))"

class Repro(torch.nn.Module):
    def forward(self, arg150_1: "i64[8]", mm: "f32[8192, 2]", arg0_1: "i64[8, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        ne_scalar: "b8[8]" = torch.ops.aten.ne.Scalar(arg150_1, -100)
        view_default: "f32[8, 1024, 2]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        iota_default: "i64[8]" = torch.ops.prims.iota.default(8, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        iota_default_1: "i32[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        ne_scalar_1: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg0_1, 0);  arg0_1 = None
        convert_element_type_default: "i32[8, 1024]" = torch.ops.prims.convert_element_type.default(ne_scalar_1, torch.int32);  ne_scalar_1 = None
        mul_tensor: "i32[8, 1024]" = torch.ops.aten.mul.Tensor(iota_default_1, convert_element_type_default);  iota_default_1 = convert_element_type_default = None
        argmax_default: "i64[8]" = torch.ops.aten.argmax.default(mul_tensor, -1);  mul_tensor = None
        index_tensor: "f32[8, 2]" = torch.ops.aten.index.Tensor(view_default, [iota_default, argmax_default]);  view_default = iota_default = argmax_default = None
        amax_default: "f32[8, 1]" = torch.ops.aten.amax.default(index_tensor, [1], True)
        sub_tensor: "f32[8, 2]" = torch.ops.aten.sub.Tensor(index_tensor, amax_default);  index_tensor = amax_default = None
        exp_default: "f32[8, 2]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[8, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[8, 2]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar_2: "b8[8]" = torch.ops.aten.ne.Scalar(arg150_1, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[8]" = torch.ops.aten.where.self(ne_scalar_2, arg150_1, full_default);  ne_scalar_2 = full_default = None
        unsqueeze_default: "i64[8, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[8, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[8]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[8]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[8]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  ne_scalar = neg_default = full_default_1 = None
        sum_default: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        ne_scalar_3: "b8[8]" = torch.ops.aten.ne.Scalar(arg150_1, -100);  arg150_1 = None
        sum_default_1: "i64[]" = torch.ops.aten.sum.default(ne_scalar_3);  ne_scalar_3 = None
        convert_element_type_default_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_1, torch.float32);  sum_default_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default, convert_element_type_default_1);  sum_default = convert_element_type_default_1 = None
        return div_tensor



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
