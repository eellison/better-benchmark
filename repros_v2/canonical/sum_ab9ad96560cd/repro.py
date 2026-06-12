"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_train
Pattern hash: ab9ad96560cd
Shape hash: d91f9612
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
    def forward(self, arg0_1: "bf16[4, 2048, 92]", arg1_1: "bf16[4, 2048, 92]"):
        # No stacktrace found for following nodes
        le: "b8[4, 2048, 92]" = torch.ops.aten.le.Scalar(arg0_1, 0);  arg0_1 = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[4, 2048, 92]" = torch.ops.aten.where.self(le, full, arg1_1);  le = arg1_1 = None
        sum_1: "bf16[2048]" = torch.ops.aten.sum.dim_IntList(where, [0, 2])
        convert_element_type: "f32[2048]" = torch.ops.prims.convert_element_type.default(sum_1, torch.float32);  sum_1 = None
        return (full, where, convert_element_type)



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
