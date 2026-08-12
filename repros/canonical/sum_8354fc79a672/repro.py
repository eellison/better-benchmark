"""
Standalone repro captured via capture_hook.
Label: hf_OPTForCausalLM_train
Pattern hash: 8354fc79a672
Shape hash: f696cede
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
    def forward(self, arg0_1: "bf16[4, 12, 2048, 64]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        permute: "bf16[4, 2048, 12, 64]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        view: "bf16[4, 2048, 768]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        mul: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(view, 0.125);  view = None
        view_1: "bf16[8192, 768]" = torch.ops.aten.view.default(mul, _shape_param_1);  mul = _shape_param_1 = None
        permute_1: "bf16[768, 8192]" = torch.ops.aten.permute.default(view_1, [1, 0])
        sum_1: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1, [0], True, dtype = torch.float32)
        view_2: "f32[768]" = torch.ops.aten.view.default(sum_1, _shape_param_2);  sum_1 = _shape_param_2 = None
        convert_element_type: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        convert_element_type_1: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        return (view_1, permute_1, convert_element_type_1)



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
