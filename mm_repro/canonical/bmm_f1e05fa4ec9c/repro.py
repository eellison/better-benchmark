"""
Standalone repro captured via capture_hook.
Label: repros/models/timm/train/swin_base_patch4_window7_224/full_graph_001.py
Pattern hash: f1e05fa4ec9c
Shape hash: 6e4e4947
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
    def forward(self, arg0_1: "bf16[4096, 49, 32]", arg1_1: "bf16[4096, 32, 49]", arg2_1: "bf16[4096, 49, 49]", arg3_1: "i64[49, 49]", arg4_1: "f32[169, 32]", arg5_1: "f32[128, 32, 49, 1]", arg6_1: "f32[128, 32, 49, 1]", arg7_1: "bf16[4096, 32, 49]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        full: "f32[169, 32]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        bmm: "bf16[4096, 49, 49]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[128, 32, 49, 49]" = torch.ops.aten.view.default(bmm, _shape_param_1);  bmm = _shape_param_1 = None
        convert_element_type: "f32[128, 32, 49, 49]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "bf16[128, 32, 49, 49]" = torch.ops.aten.view.default(arg2_1, _shape_param_2);  arg2_1 = _shape_param_2 = None
        view_2: "i64[2401]" = torch.ops.aten.view.default(arg3_1, [-1]);  arg3_1 = None
        index: "f32[2401, 32]" = torch.ops.aten.index.Tensor(arg4_1, [view_2]);  arg4_1 = None
        view_3: "f32[49, 49, 32]" = torch.ops.aten.view.default(index, _shape_param_3);  index = _shape_param_3 = None
        permute: "f32[32, 49, 49]" = torch.ops.aten.permute.default(view_3, [2, 0, 1]);  view_3 = None
        clone: "f32[32, 49, 49]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        unsqueeze: "f32[1, 32, 49, 49]" = torch.ops.aten.unsqueeze.default(clone, 0);  clone = None
        add: "f32[128, 32, 49, 49]" = torch.ops.aten.add.Tensor(view_1, unsqueeze);  view_1 = unsqueeze = None
        sub: "f32[128, 32, 49, 49]" = torch.ops.aten.sub.Tensor(add, arg5_1);  add = arg5_1 = None
        exp: "f32[128, 32, 49, 49]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[128, 32, 49, 49]" = torch.ops.aten.div.Tensor(exp, arg6_1);  exp = arg6_1 = None
        mul: "f32[128, 32, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type, div);  convert_element_type = None
        sum_1: "f32[128, 32, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul, [-1], True)
        neg: "f32[128, 32, 49, 49]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[128, 32, 49, 49]" = torch.ops.prims.fma.default(neg, sum_1, mul);  neg = sum_1 = mul = None
        convert_element_type_1: "bf16[128, 32, 49, 49]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16)
        sum_2: "f32[1, 32, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma, [0], True, dtype = torch.float32);  fma = None
        squeeze: "f32[32, 49, 49]" = torch.ops.aten.squeeze.dim(sum_2, 0);  sum_2 = None
        permute_1: "f32[49, 49, 32]" = torch.ops.aten.permute.default(squeeze, [1, 2, 0]);  squeeze = None
        view_4: "f32[2401, 32]" = torch.ops.aten.view.default(permute_1, _shape_param_4);  permute_1 = _shape_param_4 = None
        index_put: "f32[169, 32]" = torch.ops.aten.index_put.default(full, [view_2], view_4, True);  full = view_2 = view_4 = None
        view_5: "bf16[4096, 49, 49]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_5);  convert_element_type_1 = _shape_param_5 = None
        bmm_1: "bf16[4096, 32, 49]" = torch.ops.aten.bmm.default(arg7_1, view_5);  arg7_1 = None
        view_6: "bf16[128, 32, 32, 49]" = torch.ops.aten.view.default(bmm_1, _shape_param_6);  bmm_1 = _shape_param_6 = None
        permute_2: "bf16[128, 32, 49, 32]" = torch.ops.aten.permute.default(view_6, [0, 1, 3, 2]);  view_6 = None
        return (index_put, view_5, permute_2)



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
