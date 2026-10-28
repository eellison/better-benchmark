"""
Standalone repro captured via capture_hook.
Label: repros/models/timm/infer/swin_base_patch4_window7_224/full_graph_000.py
Pattern hash: 11911ec6f17b
Shape hash: 48a128cd
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
    def forward(self, arg0_1: "bf16[32768, 56, 32]", arg1_1: "bf16[32768, 32, 56]", arg2_1: "i64[49, 49]", arg3_1: "bf16[169, 4]", arg4_1: "bf16[64, 49, 49]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        bmm: "bf16[32768, 56, 56]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        slice_1: "bf16[32768, 49, 56]" = torch.ops.aten.slice.Tensor(bmm, 1, 0, -7);  bmm = None
        slice_2: "bf16[32768, 49, 49]" = torch.ops.aten.slice.Tensor(slice_1, 2, 0, -7);  slice_1 = None
        view: "bf16[8192, 4, 49, 49]" = torch.ops.aten.view.default(slice_2, _shape_param_0);  slice_2 = _shape_param_0 = None
        view_1: "i64[2401]" = torch.ops.aten.view.default(arg2_1, [-1]);  arg2_1 = None
        index: "bf16[2401, 4]" = torch.ops.aten.index.Tensor(arg3_1, [view_1]);  arg3_1 = view_1 = None
        view_2: "bf16[49, 49, 4]" = torch.ops.aten.view.default(index, _shape_param_1);  index = _shape_param_1 = None
        permute: "bf16[4, 49, 49]" = torch.ops.aten.permute.default(view_2, [2, 0, 1]);  view_2 = None
        clone: "bf16[4, 49, 49]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        unsqueeze: "bf16[1, 4, 49, 49]" = torch.ops.aten.unsqueeze.default(clone, 0);  clone = None
        add: "bf16[8192, 4, 49, 49]" = torch.ops.aten.add.Tensor(view, unsqueeze);  view = unsqueeze = None
        view_3: "bf16[128, 64, 4, 49, 49]" = torch.ops.aten.view.default(add, _shape_param_2);  add = _shape_param_2 = None
        unsqueeze_1: "bf16[64, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg4_1, 1);  arg4_1 = None
        unsqueeze_2: "bf16[1, 64, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 0);  unsqueeze_1 = None
        add_1: "bf16[128, 64, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_3, unsqueeze_2);  view_3 = unsqueeze_2 = None
        view_4: "bf16[8192, 4, 49, 49]" = torch.ops.aten.view.default(add_1, _shape_param_3);  add_1 = _shape_param_3 = None
        convert_element_type: "f32[8192, 4, 49, 49]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        amax: "f32[8192, 4, 49, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub: "f32[8192, 4, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[8192, 4, 49, 49]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[8192, 4, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[8192, 4, 49, 49]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[8192, 4, 49, 49]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        expand: "bf16[8192, 4, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_1, _shape_param_4);  convert_element_type_1 = _shape_param_4 = None
        view_5: "bf16[32768, 49, 49]" = torch.ops.aten.view.default(expand, _shape_param_5);  expand = _shape_param_5 = None
        constant_pad_nd: "bf16[32768, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_5, _shape_param_6);  view_5 = _shape_param_6 = None
        return constant_pad_nd



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
