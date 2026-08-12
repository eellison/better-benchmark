"""
Standalone repro captured via capture_hook.
Label: hf_OPTForCausalLM_train
Pattern hash: 84a4b14c240d
Shape hash: 0955d043
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
    def forward(self, arg0_1: "f32[4, 2048, 768]", arg1_1: "b8[8192, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "f32[8192, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "bf16[8192, 768]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16)
        convert_element_type_1: "bf16[8192, 768]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        mul: "bf16[8192, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "bf16[8192, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        permute: "bf16[768, 8192]" = torch.ops.aten.permute.default(mul_1, [1, 0])
        sum_1: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1, [0], True, dtype = torch.float32)
        view_1: "f32[768]" = torch.ops.aten.view.default(sum_1, _shape_param_1);  sum_1 = _shape_param_1 = None
        convert_element_type_2: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_3: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        return (view, mul_1, permute, convert_element_type_3)



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
