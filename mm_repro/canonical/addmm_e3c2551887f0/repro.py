"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/YituTechConvBert/full_graph_000.py
Pattern hash: e3c2551887f0
Shape hash: 31f0a3a9
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
    def forward(self, arg0_1: "bf16[32, 384, 512]", arg1_1: "f32[384, 1]", arg2_1: "f32[384]", arg3_1: "f32[384, 768]", arg4_1: "bf16[16384, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        add: "f32[32, 384, 512]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "bf16[32, 384, 512]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        convert_element_type_1: "bf16[384]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        convert_element_type_2: "bf16[384, 768]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        permute: "bf16[768, 384]" = torch.ops.aten.permute.default(convert_element_type_2, [1, 0]);  convert_element_type_2 = None
        addmm: "bf16[16384, 384]" = torch.ops.aten.addmm.default(convert_element_type_1, arg4_1, permute);  convert_element_type_1 = arg4_1 = permute = None
        view: "bf16[32, 512, 384]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        permute_1: "bf16[32, 512, 384]" = torch.ops.aten.permute.default(convert_element_type, [0, 2, 1]);  convert_element_type = None
        mul: "bf16[32, 512, 384]" = torch.ops.aten.mul.Tensor(permute_1, view);  permute_1 = view = None
        clone: "bf16[32, 512, 384]" = torch.ops.aten.clone.default(mul, memory_format = torch.contiguous_format);  mul = None
        view_1: "bf16[16384, 384]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        return view_1



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
