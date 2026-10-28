"""
Standalone repro captured via capture_hook.
Label: repros/models/timm/train/swin_base_patch4_window7_224/full_graph_001.py
Pattern hash: 370d6447b18e
Shape hash: 4522e7ab
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
    def forward(self, arg0_1: "bf16[401408, 128]", arg1_1: "bf16[128, 128]", arg2_1: "bf16[32768, 32, 49]", arg3_1: "bf16[32768, 56, 56]", arg4_1: "i64[49, 49]", arg5_1: "f32[169, 4]", arg6_1: "f32[64, 49, 49]", arg7_1: "f32[8192, 4, 49, 1]", arg8_1: "f32[8192, 4, 49, 1]", arg9_1: "bf16[32768, 49, 32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # No stacktrace found for following nodes
        mm: "bf16[401408, 128]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        view_1: "bf16[8192, 49, 4, 32]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute: "bf16[8192, 4, 49, 32]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        clone: "bf16[8192, 4, 49, 32]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_2: "bf16[32768, 49, 32]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        bmm: "bf16[32768, 49, 49]" = torch.ops.aten.bmm.default(view_2, arg2_1);  view_2 = arg2_1 = None
        view_3: "bf16[8192, 4, 49, 49]" = torch.ops.aten.view.default(bmm, _shape_param_3);  bmm = _shape_param_3 = None
        convert_element_type: "f32[8192, 4, 49, 49]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        slice_1: "bf16[32768, 49, 56]" = torch.ops.aten.slice.Tensor(arg3_1, 1, 0, -7);  arg3_1 = None
        slice_2: "bf16[32768, 49, 49]" = torch.ops.aten.slice.Tensor(slice_1, 2, 0, -7);  slice_1 = None
        view_4: "bf16[8192, 4, 49, 49]" = torch.ops.aten.view.default(slice_2, _shape_param_4);  slice_2 = _shape_param_4 = None
        view_5: "i64[2401]" = torch.ops.aten.view.default(arg4_1, [-1]);  arg4_1 = None
        index: "f32[2401, 4]" = torch.ops.aten.index.Tensor(arg5_1, [view_5]);  arg5_1 = view_5 = None
        view_6: "f32[49, 49, 4]" = torch.ops.aten.view.default(index, _shape_param_5);  index = _shape_param_5 = None
        permute_1: "f32[4, 49, 49]" = torch.ops.aten.permute.default(view_6, [2, 0, 1]);  view_6 = None
        clone_1: "f32[4, 49, 49]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        unsqueeze: "f32[1, 4, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_1, 0);  clone_1 = None
        add: "f32[8192, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_4, unsqueeze);  view_4 = unsqueeze = None
        view_7: "f32[128, 64, 4, 49, 49]" = torch.ops.aten.view.default(add, _shape_param_6);  add = _shape_param_6 = None
        unsqueeze_1: "f32[64, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg6_1, 1);  arg6_1 = None
        unsqueeze_2: "f32[1, 64, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 0);  unsqueeze_1 = None
        add_1: "f32[128, 64, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_7, unsqueeze_2);  view_7 = unsqueeze_2 = None
        view_8: "f32[8192, 4, 49, 49]" = torch.ops.aten.view.default(add_1, _shape_param_7);  add_1 = _shape_param_7 = None
        sub: "f32[8192, 4, 49, 49]" = torch.ops.aten.sub.Tensor(view_8, arg7_1);  view_8 = arg7_1 = None
        exp: "f32[8192, 4, 49, 49]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[8192, 4, 49, 49]" = torch.ops.aten.div.Tensor(exp, arg8_1);  exp = arg8_1 = None
        mul: "f32[8192, 4, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type, div);  convert_element_type = None
        sum_1: "f32[8192, 4, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul, [-1], True)
        neg: "f32[8192, 4, 49, 49]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[8192, 4, 49, 49]" = torch.ops.prims.fma.default(neg, sum_1, mul);  neg = sum_1 = mul = None
        convert_element_type_1: "bf16[8192, 4, 49, 49]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16)
        view_9: "bf16[32768, 49, 49]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_8);  convert_element_type_1 = _shape_param_8 = None
        bmm_1: "bf16[32768, 49, 32]" = torch.ops.aten.bmm.default(view_9, arg9_1);  arg9_1 = None
        view_10: "bf16[8192, 4, 49, 32]" = torch.ops.aten.view.default(bmm_1, _shape_param_9);  bmm_1 = _shape_param_9 = None
        mul_1: "bf16[8192, 4, 49, 32]" = torch.ops.aten.mul.Tensor(view_10, 0.1767766952966369);  view_10 = None
        return (fma, view_9, mul_1)



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
