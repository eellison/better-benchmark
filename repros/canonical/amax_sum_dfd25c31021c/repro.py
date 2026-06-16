"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: dfd25c31021c
Shape hash: de033194
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
    def forward(self, arg0_1: "bf16[16, 2]"):
        # No stacktrace found for following nodes
        convert_element_type: "f32[16, 2]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        amax: "f32[16, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub: "f32[16, 2]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[16, 2]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[16, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True);  exp = None
        log: "f32[16, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[16, 2]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        return sub_1



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
