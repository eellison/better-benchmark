"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/YituTechConvBert/full_graph_001.py
Pattern hash: 5e226d216884
Shape hash: ff165bfc
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
    def forward(self, arg0_1: "bf16[16384, 768]", arg1_1: "bf16[768, 768]", arg2_1: "f32[54]", arg3_1: "bf16[16384, 54]", arg4_1: "f32[98304, 1, 1]", arg5_1: "f32[98304, 1, 1]", arg6_1: "i64[9, 512, 1, 1]", arg7_1: "i64[1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12):
        # No stacktrace found for following nodes
        mm: "bf16[16384, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        view_1: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        slice_1: "bf16[32, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_1, 2, 6, 12);  view_1 = None
        view_2: "bf16[16384, 384]" = torch.ops.aten.view.default(slice_1, _shape_param_2);  slice_1 = _shape_param_2 = None
        clone: "bf16[16384, 384]" = torch.ops.aten.clone.default(view_2, memory_format = torch.contiguous_format);  view_2 = None
        view_3: "bf16[98304, 64, 1]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        convert_element_type: "bf16[54]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        view_4: "bf16[32, 512, 54]" = torch.ops.aten.view.default(arg3_1, _shape_param_4);  arg3_1 = _shape_param_4 = None
        add: "bf16[32, 512, 54]" = torch.ops.aten.add.Tensor(view_4, convert_element_type);  view_4 = convert_element_type = None
        view_5: "bf16[98304, 9, 1]" = torch.ops.aten.view.default(add, _shape_param_5);  add = _shape_param_5 = None
        convert_element_type_1: "f32[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        sub: "f32[98304, 9, 1]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg4_1);  convert_element_type_1 = arg4_1 = None
        exp: "f32[98304, 9, 1]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[98304, 9, 1]" = torch.ops.aten.div.Tensor(exp, arg5_1);  exp = arg5_1 = None
        convert_element_type_2: "bf16[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        expand: "bf16[98304, 9, 1]" = torch.ops.aten.expand.default(convert_element_type_2, _shape_param_6);  convert_element_type_2 = _shape_param_6 = None
        permute: "bf16[98304, 1, 9]" = torch.ops.aten.permute.default(expand, [0, 2, 1]);  expand = None
        convert_element_type_3: "f32[98304, 64, 1]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        convert_element_type_4: "f32[98304, 1, 9]" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        mul: "f32[98304, 64, 9]" = torch.ops.aten.mul.Tensor(convert_element_type_3, convert_element_type_4);  convert_element_type_3 = convert_element_type_4 = None
        convert_element_type_5: "bf16[98304, 64, 9]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        view_6: "bf16[32, 512, 384, 9]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_7);  convert_element_type_5 = _shape_param_7 = None
        view_7: "bf16[32, 512, 3456]" = torch.ops.aten.view.default(view_6, _shape_param_8);  view_6 = _shape_param_8 = None
        permute_1: "bf16[32, 3456, 512]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        convert_element_type_6: "f32[32, 3456, 512]" = torch.ops.prims.convert_element_type.default(permute_1, torch.float32);  permute_1 = None
        view_8: "f32[32, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(convert_element_type_6, _shape_param_9);  convert_element_type_6 = _shape_param_9 = None
        permute_2: "f32[32, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_8, [0, 1, 2, 4, 3, 5]);  view_8 = None
        full: "f32[32, 384, 520, 1]" = torch.ops.aten.full.default(_shape_param_10, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_10 = None
        index_put: "f32[32, 384, 520, 1]" = torch.ops.aten.index_put.default(full, [None, None, arg6_1, arg7_1], permute_2, True);  full = arg6_1 = arg7_1 = permute_2 = None
        constant_pad_nd: "f32[32, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(index_put, _shape_param_11, 0.0);  index_put = _shape_param_11 = None
        convert_element_type_7: "bf16[32, 384, 512, 1]" = torch.ops.prims.convert_element_type.default(constant_pad_nd, torch.bfloat16);  constant_pad_nd = None
        squeeze: "bf16[32, 384, 512]" = torch.ops.aten.squeeze.dim(convert_element_type_7, -1);  convert_element_type_7 = None
        permute_3: "bf16[32, 512, 384]" = torch.ops.aten.permute.default(squeeze, [0, 2, 1]);  squeeze = None
        clone_1: "bf16[32, 512, 384]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_9: "bf16[16384, 384]" = torch.ops.aten.view.default(clone_1, _shape_param_12);  clone_1 = _shape_param_12 = None
        permute_4: "bf16[384, 16384]" = torch.ops.aten.permute.default(view_9, [1, 0])
        return (view_9, permute_4)



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
