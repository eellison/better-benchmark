"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_train
Pattern hash: d3451075c4bb
Shape hash: c78a05f8
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
    def forward(self, arg0_1: "bf16[8192, 4096]", arg1_1: "i64[99]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.5)
        mul_1: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.7071067811865476);  convert_element_type = None
        erf: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        convert_element_type_1: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg1_1, 56);  arg1_1 = None
        inductor_random: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        convert_element_type_2: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_2, 0.1);  convert_element_type_2 = None
        mul_3: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt, convert_element_type_1);  convert_element_type_1 = None
        mul_4: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_3, 1.1111111111111112);  mul_3 = None
        view_1: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_4, _shape_param_2);  mul_4 = _shape_param_2 = None
        return (gt, view_1)



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
