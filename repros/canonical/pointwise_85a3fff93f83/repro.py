"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train
Pattern hash: 85a3fff93f83
Shape hash: 96064e9c
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
    def forward(self, arg0_1: "bf16[4096, 1024]", arg1_1: "bf16[4096, 1024]", arg2_1: "i64[84]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        mul: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view, 0.5)
        convert_element_type: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view, torch.float32)
        pow_1: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 3.0);  convert_element_type = None
        mul_1: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view, mul_1);  view = mul_1 = None
        mul_2: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add, 0.7978845608028654);  add = None
        tanh: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_2);  mul_2 = None
        add_1: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_3: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul, add_1);  mul = add_1 = None
        view_1: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        mul_4: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_3, view_1);  mul_3 = view_1 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg2_1, 81);  arg2_1 = None
        inductor_random: "f32[32, 128, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_2, inductor_lookup_seed, 'rand');  _shape_param_2 = inductor_lookup_seed = None
        gt: "b8[32, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul_5: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(gt, mul_4);  mul_4 = None
        mul_6: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_5, 1.1111111111111112);  mul_5 = None
        convert_element_type_1: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        view_2: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  convert_element_type_1 = _shape_param_3 = None
        return (gt, view_2)



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
