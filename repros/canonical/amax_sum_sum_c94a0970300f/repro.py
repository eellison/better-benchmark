"""
Standalone repro captured via capture_hook.
Label: hf_MegatronBertForCausalLM_infer
Pattern hash: c94a0970300f
Shape hash: 4137e900
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
    def forward(self, arg0_1: "i64[16, 512]", arg1_1: "bf16[8192, 29056]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        constant_pad_nd: "i64[16, 513]" = torch.ops.aten.constant_pad_nd.default(arg0_1, [0, 1], -100.0);  arg0_1 = None
        slice_1: "i64[16, 512]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone: "i64[16, 512]" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None
        view: "i64[8192]" = torch.ops.aten.view.default(clone, [-1]);  clone = None
        ne: "b8[8192]" = torch.ops.aten.ne.Scalar(view, -100)
        view_1: "bf16[16, 512, 29056]" = torch.ops.aten.view.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None
        convert_element_type: "f32[16, 512, 29056]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32)
        view_2: "f32[8192, 29056]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        amax: "f32[8192, 1]" = torch.ops.aten.amax.default(view_2, [1], True)
        sub: "f32[8192, 29056]" = torch.ops.aten.sub.Tensor(view_2, amax);  view_2 = amax = None
        exp: "f32[8192, 29056]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[8192, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[8192, 29056]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        ne_1: "b8[8192]" = torch.ops.aten.ne.Scalar(view, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[8192]" = torch.ops.aten.where.self(ne_1, view, full);  ne_1 = full = None
        unsqueeze: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[8192, 1]" = torch.ops.aten.gather.default(sub_1, 1, unsqueeze);  sub_1 = unsqueeze = None
        squeeze: "f32[8192]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[8192]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[8192]" = torch.ops.aten.where.self(ne, neg, full_1);  ne = neg = full_1 = None
        sum_2: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        ne_2: "b8[8192]" = torch.ops.aten.ne.Scalar(view, -100);  view = None
        sum_3: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_3, torch.float32);  sum_3 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_2, convert_element_type_1);  sum_2 = convert_element_type_1 = None
        return (view_1, div)



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
