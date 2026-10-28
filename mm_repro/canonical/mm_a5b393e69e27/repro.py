"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/MobileBertForMaskedLM/full_graph_001.py
Pattern hash: a5b393e69e27
Shape hash: 54215804
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
    def forward(self, arg0_1: "bf16[512, 32768]", arg1_1: "bf16[32768, 30528]", arg2_1: "bf16[512, 32768]", arg3_1: "bf16[32768, 512]", arg4_1: "bf16[512, 32768]", arg5_1: "bf16[32768, 128]"):
        # No stacktrace found for following nodes
        mm: "bf16[512, 30528]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        slice_1: "bf16[512, 30522]" = torch.ops.aten.slice.Tensor(mm, 1, 0, -6);  mm = None
        convert_element_type: "f32[512, 30522]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        slice_2: "f32[128, 30522]" = torch.ops.aten.slice.Tensor(convert_element_type, 0, 0, 128)
        slice_3: "f32[384, 30522]" = torch.ops.aten.slice.Tensor(convert_element_type, 0, 128, 512);  convert_element_type = None
        permute: "f32[30522, 128]" = torch.ops.aten.permute.default(slice_2, [1, 0]);  slice_2 = None
        mm_1: "bf16[512, 512]" = torch.ops.aten.mm.default(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
        convert_element_type_1: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        mm_2: "bf16[512, 128]" = torch.ops.aten.mm.default(arg4_1, arg5_1);  arg4_1 = arg5_1 = None
        convert_element_type_2: "f32[512, 128]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        return (slice_3, permute, convert_element_type_1, convert_element_type_2)



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
