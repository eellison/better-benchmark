"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/M2M100ForConditionalGeneration/full_graph_009.py
Pattern hash: 1aab21d778e6
Shape hash: a950af5a
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
    def forward(self, arg0_1: "f32[64, 128, 1024]", arg1_1: "b8[64, 128, 1024]", arg2_1: "bf16[1024, 4096]", arg3_1: "b8[64, 128, 4096]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        mul: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        view: "bf16[8192, 1024]" = torch.ops.aten.view.default(mul_1, _shape_param_0);  mul_1 = _shape_param_0 = None
        mm: "bf16[8192, 4096]" = torch.ops.aten.mm.default(view, arg2_1);  view = arg2_1 = None
        view_1: "bf16[64, 128, 4096]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[64, 128, 4096]" = torch.ops.aten.where.self(arg3_1, full, view_1);  arg3_1 = full = view_1 = None
        view_2: "bf16[8192, 4096]" = torch.ops.aten.view.default(where, _shape_param_2);  where = _shape_param_2 = None
        permute: "bf16[4096, 8192]" = torch.ops.aten.permute.default(view_2, [1, 0])
        return (view_2, permute)



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
