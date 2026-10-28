"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/MobileBertForMaskedLM/full_graph_001.py
Pattern hash: 687e08371fcc
Shape hash: 36277280
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
    def forward(self, arg0_1: "bf16[1024, 128, 128]", arg1_1: "bf16[1024, 128, 32]", arg2_1: "bf16[128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        bmm: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        mul: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(view, 0.4204482076268573);  view = None
        permute: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(mul, [0, 2, 1, 3]);  mul = None
        clone: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        view_2: "bf16[32768, 128]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        mm: "bf16[32768, 128]" = torch.ops.aten.mm.default(view_2, arg2_1);  arg2_1 = None
        permute_1: "bf16[128, 32768]" = torch.ops.aten.permute.default(view_2, [1, 0])
        view_3: "bf16[256, 128, 128]" = torch.ops.aten.view.default(mm, _shape_param_3);  mm = _shape_param_3 = None
        convert_element_type: "f32[256, 128, 128]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        return (view_2, permute_1, convert_element_type)



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
