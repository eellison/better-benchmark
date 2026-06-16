"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_train
Pattern hash: 07f3b9d27881
Shape hash: 363c89c3
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
    def forward(self, arg0_1: "bf16[8192, 2048]", arg1_1: "b8[8, 1024, 2048]", arg2_1: "b8[8, 1024, 2048]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_1: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        mul: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[8, 1024, 2048]" = torch.ops.aten.where.self(arg2_1, full, mul_1);  arg2_1 = mul_1 = None
        view_1: "bf16[8192, 2048]" = torch.ops.aten.view.default(where, _shape_param_1);  where = _shape_param_1 = None
        permute: "bf16[2048, 8192]" = torch.ops.aten.permute.default(view_1, [1, 0])
        return (full, view_1, permute)



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
