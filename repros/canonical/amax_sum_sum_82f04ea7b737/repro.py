"""
Standalone repro captured via capture_hook.
Label: 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_train
Pattern hash: 82f04ea7b737
Shape hash: a0e86354
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
    def forward(self, arg0_1: "bf16[2048, 8008]", arg1_1: "f32[1, 8008]", arg2_1: "i64[16, 128]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[16, 128, 8008]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        add: "f32[16, 128, 8008]" = torch.ops.aten.add.Tensor(view, arg1_1);  view = arg1_1 = None
        view_1: "f32[2048, 8008]" = torch.ops.aten.view.default(add, _shape_param_1);  _shape_param_1 = None
        view_2: "i64[2048]" = torch.ops.aten.view.default(arg2_1, [-1]);  arg2_1 = None
        amax: "f32[2048, 1]" = torch.ops.aten.amax.default(view_1, [1], True)
        sub: "f32[2048, 8008]" = torch.ops.aten.sub.Tensor(view_1, amax);  view_1 = None
        exp: "f32[2048, 8008]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[2048, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[2048, 8008]" = torch.ops.aten.sub.Tensor(sub, log);  sub = None
        ne: "b8[2048]" = torch.ops.aten.ne.Scalar(view_2, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[2048]" = torch.ops.aten.where.self(ne, view_2, full);  view_2 = full = None
        unsqueeze: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[2048, 1]" = torch.ops.aten.gather.default(sub_1, 1, unsqueeze);  sub_1 = unsqueeze = None
        squeeze: "f32[2048]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[2048]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[2048]" = torch.ops.aten.where.self(ne, neg, full_1);  neg = full_1 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type);  sum_3 = None
        return (add, amax, log, convert_element_type, div)



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
