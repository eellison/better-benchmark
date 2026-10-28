"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/GPTNeoForCausalLM/full_graph_001.py
Pattern hash: d0b08749d87e
Shape hash: 2dfeae8f
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
    def forward(self, arg0_1: "bf16[512, 128, 128]", arg1_1: "bf16[512, 128, 128]", arg2_1: "f32[32, 16, 128, 128]", arg3_1: "b8[1, 1, 2048, 2048]", arg4_1: "bf16[512, 128, 128]", arg5_1: "bf16[512, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        bmm: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        convert_element_type: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  convert_element_type = None
        sum_1: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul, [-1], True)
        neg: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg2_1);  arg2_1 = None
        fma: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg, sum_1, mul);  neg = sum_1 = mul = None
        convert_element_type_1: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_1: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg3_1, 2, 0, 128);  arg3_1 = None
        slice_2: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 128);  slice_1 = None
        where: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_2, convert_element_type_1, full);  slice_2 = convert_element_type_1 = full = None
        view_1: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where, _shape_param_1);  where = _shape_param_1 = None
        bmm_1: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg4_1, view_1);  arg4_1 = None
        bmm_2: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_1, arg5_1);  view_1 = arg5_1 = None
        view_2: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_1, _shape_param_2);  bmm_1 = _shape_param_2 = None
        view_3: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_3);  bmm_2 = _shape_param_3 = None
        convert_element_type_2: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        permute: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_2, [0, 1, 3, 2]);  convert_element_type_2 = None
        convert_element_type_3: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute, torch.bfloat16);  permute = None
        convert_element_type_4: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        permute_1: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_3, [0, 2, 1, 3]);  convert_element_type_3 = None
        view_4: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_1, _shape_param_4);  permute_1 = _shape_param_4 = None
        permute_2: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_4, [0, 2, 1, 3]);  convert_element_type_4 = None
        clone: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_5: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone, _shape_param_5);  clone = _shape_param_5 = None
        clone_1: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_4, memory_format = torch.contiguous_format);  view_4 = None
        view_6: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_1, _shape_param_6);  clone_1 = _shape_param_6 = None
        permute_3: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_6, [1, 0])
        view_7: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_5, _shape_param_7);  view_5 = _shape_param_7 = None
        permute_4: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_7, [1, 0])
        return (view_6, permute_3, view_7, permute_4)



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
