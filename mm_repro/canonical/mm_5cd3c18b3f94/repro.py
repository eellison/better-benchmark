"""
Standalone repro captured via capture_hook.
Label: repros/models/timm/train/swin_base_patch4_window7_224/full_graph_001.py
Pattern hash: 5cd3c18b3f94
Shape hash: 8afdfb1f
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
    def forward(self, arg0_1: "bf16[25088, 1536]", arg1_1: "bf16[1536, 512]", arg2_1: "f32[512]", arg3_1: "bf16[128, 14, 14, 512]", arg4_1: "f32[128, 14, 14, 1]", arg5_1: "f32[128, 14, 14, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        mm: "bf16[25088, 512]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        view_2: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        permute: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2, 4, 5]);  view_2 = None
        clone: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_3: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        mul: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_3, arg2_1);  arg2_1 = None
        mul_1: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul, 512)
        convert_element_type_1: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        sub: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg4_1);  convert_element_type_1 = arg4_1 = None
        mul_2: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub, arg5_1);  sub = arg5_1 = None
        mul_3: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul, mul_2)
        mul_4: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_3, mul_2);  view_3 = mul_2 = None
        return (mul, mul_1, mul_3, mul_4)



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
