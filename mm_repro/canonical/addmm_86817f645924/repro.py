"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/infer/AllenaiLongformerBase/full_graph_003.py
Pattern hash: 86817f645924
Shape hash: c97988f7
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
    def forward(self, arg0_1: "bf16[50265]", arg1_1: "bf16[50265, 768]", arg2_1: "bf16[8192, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        full: "bf16[7]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        cat: "bf16[50272]" = torch.ops.aten.cat.default([arg0_1, full]);  arg0_1 = full = None
        permute: "bf16[768, 50265]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        constant_pad_nd: "bf16[768, 50272]" = torch.ops.aten.constant_pad_nd.default(permute, _shape_param_1);  permute = _shape_param_1 = None
        addmm: "bf16[8192, 50272]" = torch.ops.aten.addmm.default(cat, arg2_1, constant_pad_nd);  cat = arg2_1 = constant_pad_nd = None
        slice_1: "bf16[8192, 50265]" = torch.ops.aten.slice.Tensor(addmm, 1, 0, -7);  addmm = None
        view: "bf16[8, 1024, 50265]" = torch.ops.aten.view.default(slice_1, _shape_param_2);  slice_1 = _shape_param_2 = None
        view_1: "bf16[8192, 50265]" = torch.ops.aten.view.default(view, _shape_param_3);  _shape_param_3 = None
        convert_element_type: "f32[8192, 50265]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        return (view, convert_element_type)



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
