"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train
Pattern hash: a72724ce25f0
Shape hash: 206932cb
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
    def forward(self, arg0_1: "bf16[128, 256, 48, 48]", arg1_1: "bf16[128, 256, 1, 1]", arg2_1: "bf16[128, 256, 1, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        mul: "bf16[128, 256, 48, 48]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        expand: "bf16[128, 256, 48, 48]" = torch.ops.aten.expand.default(arg2_1, _shape_param_0);  arg2_1 = _shape_param_0 = None
        div: "bf16[128, 256, 48, 48]" = torch.ops.aten.div.Scalar(expand, 2304);  expand = None
        add: "bf16[128, 256, 48, 48]" = torch.ops.aten.add.Tensor(mul, div);  mul = div = None
        sum_1: "bf16[256]" = torch.ops.aten.sum.dim_IntList(add, [0, 2, 3])
        convert_element_type: "f32[256]" = torch.ops.prims.convert_element_type.default(sum_1, torch.float32);  sum_1 = None
        return (add, convert_element_type)



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
