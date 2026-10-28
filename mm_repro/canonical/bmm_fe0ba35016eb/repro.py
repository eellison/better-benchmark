"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/AllenaiLongformerBase/full_graph_005.py
Pattern hash: fe0ba35016eb
Shape hash: cbc46054
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
    def forward(self, arg0_1: "bf16[384, 256, 64]", arg1_1: "bf16[384, 64, 768]", arg2_1: "b8[8, 1024, 12, 513]", arg3_1: "b8[8, 1024]", arg4_1: "bf16[8, 1024, 12, 513]", arg5_1: "f32[8, 1024, 12, 1]", arg6_1: "f32[8, 1024, 12, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        bmm: "bf16[384, 256, 768]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        squeeze: "bf16[96, 4, 256, 768]" = torch.ops.aten.squeeze.dim(view, 4);  view = None
        full: "bf16[96, 4, 256, 769]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        slice_scatter: "bf16[96, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full, squeeze, 3, 0, -1);  full = squeeze = None
        view_1: "bf16[96, 4, 196864]" = torch.ops.aten.view.default(slice_scatter, _shape_param_2);  slice_scatter = _shape_param_2 = None
        full_1: "bf16[96, 4, 197120]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        slice_scatter_1: "bf16[96, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_1, view_1, 2, 0, -256);  full_1 = view_1 = None
        view_2: "bf16[96, 4, 256, 770]" = torch.ops.aten.view.default(slice_scatter_1, _shape_param_4);  slice_scatter_1 = _shape_param_4 = None
        constant_pad_nd: "bf16[96, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_2, _shape_param_5);  view_2 = _shape_param_5 = None
        view_3: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(constant_pad_nd, _shape_param_6);  constant_pad_nd = _shape_param_6 = None
        permute: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        convert_element_type: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        mul: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_1: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute, mul);  permute = mul = None
        clone: "bf16[8, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_1, memory_format = torch.contiguous_format);  mul_1 = None
        convert_element_type_1: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(clone, torch.float32);  clone = None
        unsqueeze: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, 2);  arg3_1 = None
        unsqueeze_1: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 3);  unsqueeze = None
        full_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_1, full_2, convert_element_type_1);  unsqueeze_1 = full_2 = convert_element_type_1 = None
        convert_element_type_2: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        clone_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_2, memory_format = torch.contiguous_format);  convert_element_type_2 = None
        sub: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_1, arg5_1);  clone_1 = arg5_1 = None
        exp: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp, arg6_1);  exp = arg6_1 = None
        mul_2: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where, div);  where = div = None
        return mul_2



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
