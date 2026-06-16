"""
Standalone repro captured via capture_hook.
Label: genai_CrossEntropyForward_static
Pattern hash: 3aad2361105e
Shape hash: b899b223
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
    def forward(self, arg0_1: "i64[8192]", arg1_1: "bf16[8192, 262144]"):
        # No stacktrace found for following nodes
        ne: "b8[8192]" = torch.ops.aten.ne.Scalar(arg0_1, -100)
        convert_element_type: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        amax: "f32[8192, 1]" = torch.ops.aten.amax.default(convert_element_type, [1], True)
        sub: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[8192, 262144]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[8192, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        convert_element_type_1: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        ne_1: "b8[8192]" = torch.ops.aten.ne.Scalar(arg0_1, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[8192]" = torch.ops.aten.where.self(ne_1, arg0_1, full);  ne_1 = arg0_1 = full = None
        unsqueeze: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "bf16[8192, 1]" = torch.ops.aten.gather.default(convert_element_type_1, 1, unsqueeze);  convert_element_type_1 = unsqueeze = None
        squeeze: "bf16[8192]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[8192]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[8192]" = torch.ops.aten.where.self(ne, neg, full_1);  ne = neg = full_1 = None
        return where_1



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
