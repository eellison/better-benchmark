"""
Standalone repro captured via capture_hook.
Label: repros/models/timm/train/swin_base_patch4_window7_224/full_graph_001.py
Pattern hash: 3a3508f03d7a
Shape hash: a87f561d
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
    def forward(self, arg0_1: "bf16[401408, 384]", arg1_1: "bf16[384, 128]", arg2_1: "i64[56]", arg3_1: "f32[128]", arg4_1: "f32[128, 56, 56, 128]", arg5_1: "f32[128, 56, 56, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        mm: "bf16[401408, 128]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[8192, 49, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "f32[8192, 7, 7, 128]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        view_2: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        permute: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2, 4, 5]);  view_2 = None
        clone: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_3: "f32[128, 56, 56, 128]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        index: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(view_3, [None, None, arg2_1]);  view_3 = None
        index_1: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index, [None, arg2_1]);  index = arg2_1 = None
        mul: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(index_1, arg3_1);  arg3_1 = None
        mul_1: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul, 128)
        sum_1: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul, [3], True)
        mul_2: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = None
        sum_2: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [3], True);  mul_2 = None
        mul_3: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(arg4_1, sum_2);  sum_2 = None
        sub: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(arg5_1, sub_1);  arg5_1 = sub_1 = None
        mul_5: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(index_1, arg4_1);  arg4_1 = None
        return (index_1, mul_4, mul_5)



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
