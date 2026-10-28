"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/GPTJForCausalLM/full_graph_001.py
Pattern hash: af04f75df852
Shape hash: 56b3f831
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
    def forward(self, arg0_1: "bf16[16, 256, 128]", arg1_1: "bf16[16, 128, 128]", arg2_1: "bf16[1, 128, 1, 32, 1]", arg3_1: "bf16[1, 128, 1, 32, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # No stacktrace found for following nodes
        bmm: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        convert_element_type: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        permute: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type, [0, 1, 3, 2]);  convert_element_type = None
        convert_element_type_1: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute, torch.bfloat16);  permute = None
        permute_1: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_1, [0, 2, 1, 3]);  convert_element_type_1 = None
        slice_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1, 3, 0, 64)
        slice_2: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1, 3, 64, 256);  permute_1 = None
        expand: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        clone: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_1: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        full: "bf16[1, 128, 16, 64]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        expand_1: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg3_1, _shape_param_4);  arg3_1 = _shape_param_4 = None
        clone_1: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_2: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_5);  clone_1 = _shape_param_5 = None
        mul: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1, view_1);  view_1 = None
        view_3: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul, _shape_param_6);  mul = _shape_param_6 = None
        select: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_3, -1, 0)
        select_1: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_3, -1, 1);  view_3 = None
        neg: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select);  select = None
        slice_scatter: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full, neg, 3, 1, 9223372036854775807, 2);  neg = None
        slice_scatter_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full, select_1, 3, 0, 9223372036854775807, 2);  full = select_1 = None
        add: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter, slice_scatter_1);  slice_scatter = slice_scatter_1 = None
        mul_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1, view_2);  slice_1 = view_2 = None
        add_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add, mul_1);  add = mul_1 = None
        full_1: "bf16[1, 128, 16, 256]" = torch.ops.aten.full.default(_shape_param_7, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_7 = None
        slice_scatter_2: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_1, slice_2, 3, 64, 9223372036854775807);  slice_2 = None
        slice_scatter_3: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_1, add_1, 3, 0, 64);  full_1 = add_1 = None
        add_2: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_2, slice_scatter_3);  slice_scatter_2 = slice_scatter_3 = None
        view_4: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_2, _shape_param_8);  add_2 = _shape_param_8 = None
        view_5: "bf16[128, 4096]" = torch.ops.aten.view.default(view_4, _shape_param_9);  view_4 = _shape_param_9 = None
        permute_2: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_5, [1, 0])
        return (view_5, permute_2)



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
