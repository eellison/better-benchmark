"""
Standalone repro captured via capture_hook.
Label: hf_OPTForCausalLM_train
Pattern hash: 6e7efb08a440
Shape hash: 84318c05
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
    def forward(self, arg0_1: "i64[2]", arg1_1: "bf16[8192, 768]", arg2_1: "f32[8192, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg0_1, 1);  arg0_1 = None
        inductor_random: "f32[8192, 768]" = torch.ops.prims.inductor_random.default(_shape_param_0, inductor_lookup_seed, 'rand');  _shape_param_0 = inductor_lookup_seed = None
        convert_element_type: "bf16[8192, 768]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[8192, 768]" = torch.ops.aten.gt.Scalar(convert_element_type, 0.1);  convert_element_type = None
        mul: "bf16[8192, 768]" = torch.ops.aten.mul.Tensor(gt, arg1_1);  arg1_1 = None
        mul_1: "bf16[8192, 768]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        add: "f32[8192, 768]" = torch.ops.aten.add.Tensor(arg2_1, mul_1);  arg2_1 = mul_1 = None
        view: "f32[4, 2048, 768]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        return (gt, view)



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
