"""
Standalone repro captured via capture_hook.
Label: repros/models/torchbench/train/vgg16/full_graph_001.py
Pattern hash: 924994404cac
Shape hash: 6ecbd8b6
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
    def forward(self, arg0_1: "bf16[64, 1000]", arg1_1: "bf16[1000, 4096]", arg2_1: "b8[64, 4096]", arg3_1: "b8[64, 4096]"):
        # No stacktrace found for following nodes
        mm: "bf16[64, 4096]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "bf16[64, 4096]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        mul: "bf16[64, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type, 2.0);  convert_element_type = None
        mul_1: "bf16[64, 4096]" = torch.ops.aten.mul.Tensor(mm, mul);  mm = mul = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[64, 4096]" = torch.ops.aten.where.self(arg3_1, full, mul_1);  arg3_1 = full = mul_1 = None
        permute: "bf16[4096, 64]" = torch.ops.aten.permute.default(where, [1, 0])
        return (where, permute)



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
