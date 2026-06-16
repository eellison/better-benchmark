"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train
Pattern hash: cb474de4ede0
Shape hash: 4e4a9284
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
    def forward(self, arg0_1: "bf16[32768, 512]", arg1_1: "f32[256, 128, 512]", arg2_1: "bf16[32768, 512]", arg3_1: "bf16[32768, 512]", arg4_1: "bf16[32768, 512]", arg5_1: "f32[1, 128, 512]", arg6_1: "f32[256, 128, 512]", arg7_1: "f32[512]", arg8_1: "i64[256, 128]", arg9_1: "i64[1, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # No stacktrace found for following nodes
        view: "bf16[256, 128, 512]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[256, 128, 512]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        add: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(arg1_1, convert_element_type);  arg1_1 = convert_element_type = None
        view_1: "bf16[256, 128, 512]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        convert_element_type_1: "f32[256, 128, 512]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add, convert_element_type_1);  add = convert_element_type_1 = None
        view_2: "bf16[256, 128, 512]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        convert_element_type_2: "f32[256, 128, 512]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add_2: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add_1, convert_element_type_2);  add_1 = convert_element_type_2 = None
        sum_1: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(add_2, [0, 1], True, dtype = torch.float32)
        view_3: "f32[512]" = torch.ops.aten.view.default(sum_1, _shape_param_3);  sum_1 = _shape_param_3 = None
        view_4: "bf16[256, 128, 512]" = torch.ops.aten.view.default(arg4_1, _shape_param_4);  arg4_1 = _shape_param_4 = None
        add_3: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_4, arg5_1);  view_4 = arg5_1 = None
        add_4: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add_3, arg6_1);  add_3 = arg6_1 = None
        mul: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_2, add_4);  add_4 = None
        mul_1: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_2, arg7_1);  add_2 = arg7_1 = None
        sum_2: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul, [0, 1], True, dtype = torch.float32);  mul = None
        view_5: "f32[512]" = torch.ops.aten.view.default(sum_2, _shape_param_5);  sum_2 = _shape_param_5 = None
        convert_element_type_3: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16)
        sum_3: "f32[1, 128, 512]" = torch.ops.aten.sum.dim_IntList(mul_1, [0], True, dtype = torch.float32)
        full: "b8[256, 128, 1]" = torch.ops.aten.full.default(_shape_param_6, True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_6 = None
        full_1: "f32[2, 512]" = torch.ops.aten.full.default(_shape_param_7, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_7 = None
        _unsafe_masked_index_put_accumulate: "f32[2, 512]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_1, full, [arg8_1], mul_1);  full_1 = full = arg8_1 = mul_1 = None
        slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg9_1, 1, 0, 128);  arg9_1 = None
        ge: "b8[1, 128]" = torch.ops.aten.ge.Scalar(slice_1, 0)
        lt: "b8[1, 128]" = torch.ops.aten.lt.Scalar(slice_1, 512)
        bitwise_and: "b8[1, 128]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[1, 128]" = torch.ops.aten.ne.Scalar(slice_1, -1)
        bitwise_and_1: "b8[1, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_2: "f32[512, 512]" = torch.ops.aten.full.default(_shape_param_8, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_8 = None
        _unsafe_masked_index_put_accumulate_1: "f32[512, 512]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, unsqueeze, [slice_1], sum_3);  full_2 = unsqueeze = slice_1 = sum_3 = None
        view_6: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_9);  convert_element_type_3 = _shape_param_9 = None
        permute: "bf16[512, 32768]" = torch.ops.aten.permute.default(view_6, [1, 0])
        sum_4: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_6, [0], True, dtype = torch.float32)
        view_7: "f32[512]" = torch.ops.aten.view.default(sum_4, _shape_param_10);  sum_4 = _shape_param_10 = None
        convert_element_type_4: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_7, torch.bfloat16);  view_7 = None
        convert_element_type_5: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        return (view_3, view_5, _unsafe_masked_index_put_accumulate, _unsafe_masked_index_put_accumulate_1, view_6, permute, convert_element_type_5)



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
