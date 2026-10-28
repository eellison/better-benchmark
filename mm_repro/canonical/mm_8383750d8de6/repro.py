"""
Standalone repro captured via capture_hook.
Label: repros/models/torchbench/train/BERT_pytorch/full_graph_001.py
Pattern hash: 8383750d8de6
Shape hash: bf199367
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
    def forward(self, arg0_1: "f32[16, 2]", arg1_1: "f32[16, 2]", arg2_1: "bf16[2, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        exp: "f32[16, 2]" = torch.ops.aten.exp.default(arg0_1);  arg0_1 = None
        sum_1: "f32[16, 1]" = torch.ops.aten.sum.dim_IntList(arg1_1, [-1], True)
        mul: "f32[16, 2]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub: "f32[16, 2]" = torch.ops.aten.sub.Tensor(arg1_1, mul);  arg1_1 = mul = None
        convert_element_type: "bf16[16, 2]" = torch.ops.prims.convert_element_type.default(sub, torch.bfloat16);  sub = None
        mm: "bf16[16, 768]" = torch.ops.aten.mm.default(convert_element_type, arg2_1);  convert_element_type = arg2_1 = None
        convert_element_type_1: "f32[16, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        full: "f32[16, 128, 768]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        select_scatter: "f32[16, 128, 768]" = torch.ops.aten.select_scatter.default(full, convert_element_type_1, 1, 0);  full = convert_element_type_1 = None
        return select_scatter



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
