"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train
Pattern hash: c5765737e761
Shape hash: dfc700a0
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
    def forward(self, arg0_1: "bf16[32, 1000]", arg1_1: "b8[32, 1000, 13, 13]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[32, 1000, 1, 1]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        expand: "bf16[32, 1000, 13, 13]" = torch.ops.aten.expand.default(view, _shape_param_1);  view = _shape_param_1 = None
        div: "bf16[32, 1000, 13, 13]" = torch.ops.aten.div.Scalar(expand, 169);  expand = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[32, 1000, 13, 13]" = torch.ops.aten.where.self(arg1_1, full, div);  arg1_1 = div = None
        sum_1: "bf16[1000]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        convert_element_type: "f32[1000]" = torch.ops.prims.convert_element_type.default(sum_1, torch.float32);  sum_1 = None
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
