"""
Standalone repro captured via capture_hook.
Label: torchbench_alexnet_train
Pattern hash: cbfed5cd52de
Shape hash: 35c60c30
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
    def forward(self, arg0_1: "bf16[128, 4096]", arg1_1: "i64[2]", _shape_param_0):
        # No stacktrace found for following nodes
        relu: "bf16[128, 4096]" = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg1_1, 1);  arg1_1 = None
        inductor_random: "f32[128, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_0, inductor_lookup_seed, 'rand');  _shape_param_0 = inductor_lookup_seed = None
        convert_element_type: "bf16[128, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[128, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type, 0.5);  convert_element_type = None
        mul: "bf16[128, 4096]" = torch.ops.aten.mul.Tensor(gt, relu)
        mul_1: "bf16[128, 4096]" = torch.ops.aten.mul.Tensor(mul, 2.0);  mul = None
        le: "b8[128, 4096]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        return (gt, mul_1, le)



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
