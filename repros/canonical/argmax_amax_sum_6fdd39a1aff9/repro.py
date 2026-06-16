"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForSequenceClassification_train
Pattern hash: 6fdd39a1aff9
Shape hash: 6bcef7c8
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[4096, 2]", arg1_1: "i64[32, 128]", arg2_1: "i64[32]", arg3_1: "i64[32]", arg4_1: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[32, 128, 2]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        ne: "b8[32, 128]" = torch.ops.aten.ne.Scalar(arg1_1, 0);  arg1_1 = None
        convert_element_type: "i32[32, 128]" = torch.ops.prims.convert_element_type.default(ne, torch.int32);  ne = None
        iota: "i32[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        mul: "i32[32, 128]" = torch.ops.aten.mul.Tensor(iota, convert_element_type);  iota = convert_element_type = None
        argmax: "i64[32]" = torch.ops.aten.argmax.default(mul, -1);  mul = None
        index: "bf16[32, 2]" = torch.ops.aten.index.Tensor(view, [arg2_1, argmax]);  view = arg2_1 = None
        convert_element_type_1: "f32[32, 2]" = torch.ops.prims.convert_element_type.default(index, torch.float32)
        amax: "f32[32, 1]" = torch.ops.aten.amax.default(convert_element_type_1, [1], True)
        sub: "f32[32, 2]" = torch.ops.aten.sub.Tensor(convert_element_type_1, amax);  convert_element_type_1 = amax = None
        exp: "f32[32, 2]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[32, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[32, 2]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        convert_element_type_2: "bf16[32, 2]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_3: "f32[32, 2]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        ne_1: "b8[32]" = torch.ops.aten.ne.Scalar(arg3_1, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[32]" = torch.ops.aten.where.self(ne_1, arg3_1, full)
        unsqueeze: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[32, 1]" = torch.ops.aten.gather.default(convert_element_type_3, 1, unsqueeze);  convert_element_type_3 = unsqueeze = None
        squeeze: "f32[32]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[32]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        where_1: "f32[32]" = torch.ops.aten.where.self(ne_1, neg, arg4_1);  neg = arg4_1 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne_1);  ne_1 = None
        convert_element_type_4: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_4);  sum_3 = None
        unsqueeze_1: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, 1);  arg3_1 = None
        ne_2: "b8[32, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
        where_2: "i64[32, 1]" = torch.ops.aten.where.self(ne_2, unsqueeze_1, full);  unsqueeze_1 = full = None
        iota_1: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_1: "i64[1, 2]" = torch.ops.aten.view.default(iota_1, _shape_param_1);  iota_1 = _shape_param_1 = None
        expand: "i64[32, 2]" = torch.ops.aten.expand.default(where_2, _shape_param_2);  where_2 = _shape_param_2 = None
        eq: "b8[32, 2]" = torch.ops.aten.eq.Tensor(expand, view_1);  expand = view_1 = None
        return (argmax, index, convert_element_type_4, div, ne_2, eq)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
