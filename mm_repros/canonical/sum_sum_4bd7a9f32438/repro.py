"""
Standalone repro captured via capture_hook.
Label: XLNetLMHeadModel
Pattern hash: 4bd7a9f32438
Shape hash: 0dfc9b9d
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
    def forward(self, arg0_1: "bf16[8192, 32000]", arg1_1: "bf16[8192, 1024]", arg2_1: "bf16[256, 512, 512]", arg3_1: "bf16[256, 512, 64]", arg4_1: "bf16[16, 16, 512, 1023]", arg5_1: "bf16[16, 16, 1024, 512]", arg6_1: "bf16[256, 64, 512]", arg7_1: "bf16[256, 1024, 64]", arg8_1: "bf16[16, 16, 512, 512]", arg9_1: "bf16[256, 64, 512]", arg10_1: "bf16[256, 512, 64]", arg11_1: "bf16[1024, 16384]", arg12_1: "bf16[1024, 8192]", arg13_1: "bf16[1024, 1024]", arg14_1: "f32[512, 16, 1024]", arg15_1: "bf16[1024, 1024]", arg16_1: "bf16[1024, 1024]", arg17_1: "b8[512, 16, 1024]", arg18_1: "i64[512, 16]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28):
        # No stacktrace found for following nodes
        permute: "bf16[32000, 8192]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mm: "bf16[32000, 1024]" = torch.ops.aten.mm.default(permute, arg1_1);  permute = arg1_1 = None
        convert_element_type: "f32[32000, 1024]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        bmm: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
        view: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        permute_1: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(view, [4, 0, 1, 3, 2]);  view = None
        permute_2: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1, [4, 1, 2, 3, 0]);  permute_1 = None
        squeeze: "bf16[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2, 4);  permute_2 = None
        view_1: "bf16[16, 16, 1023, 512]" = torch.ops.aten.view.default(arg4_1, _shape_param_1);  arg4_1 = _shape_param_1 = None
        slice_scatter: "bf16[16, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(arg5_1, view_1, 2, 1, 9223372036854775807);  arg5_1 = view_1 = None
        view_2: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(slice_scatter, _shape_param_2);  slice_scatter = _shape_param_2 = None
        view_3: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.view.default(view_2, _shape_param_3);  view_2 = _shape_param_3 = None
        permute_3: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.permute.default(view_3, [0, 1, 2, 4, 3]);  view_3 = None
        view_4: "bf16[256, 512, 1024]" = torch.ops.aten.view.default(permute_3, _shape_param_4);  permute_3 = _shape_param_4 = None
        bmm_1: "bf16[256, 64, 1024]" = torch.ops.aten.bmm.default(arg6_1, view_4);  arg6_1 = None
        bmm_2: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_4, arg7_1);  view_4 = arg7_1 = None
        view_5: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.view.default(bmm_1, _shape_param_5);  bmm_1 = _shape_param_5 = None
        permute_4: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_5, [0, 1, 4, 3, 2]);  view_5 = None
        view_6: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.view.default(bmm_2, _shape_param_6);  bmm_2 = _shape_param_6 = None
        permute_5: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_6, [0, 1, 2, 4, 3]);  view_6 = None
        permute_6: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_4, [3, 0, 1, 4, 2]);  permute_4 = None
        squeeze_1: "bf16[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_6, 4);  permute_6 = None
        permute_7: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_5, [2, 0, 1, 4, 3]);  permute_5 = None
        squeeze_2: "bf16[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_7, 4);  permute_7 = None
        convert_element_type_1: "f32[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_2, torch.float32);  squeeze_2 = None
        convert_element_type_2: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.bfloat16)
        sum_1: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [0, 1], True, dtype = torch.float32);  convert_element_type_1 = None
        view_7: "f32[16, 64]" = torch.ops.aten.view.default(sum_1, _shape_param_7);  sum_1 = _shape_param_7 = None
        view_8: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.view.default(arg8_1, _shape_param_8);  arg8_1 = _shape_param_8 = None
        permute_8: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.permute.default(view_8, [0, 1, 2, 4, 3]);  view_8 = None
        view_9: "bf16[256, 512, 512]" = torch.ops.aten.view.default(permute_8, _shape_param_9);  permute_8 = _shape_param_9 = None
        bmm_3: "bf16[256, 64, 512]" = torch.ops.aten.bmm.default(arg9_1, view_9);  arg9_1 = None
        bmm_4: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_9, arg10_1);  view_9 = arg10_1 = None
        view_10: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.view.default(bmm_3, _shape_param_10);  bmm_3 = _shape_param_10 = None
        permute_9: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 1, 4, 3, 2]);  view_10 = None
        view_11: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.view.default(bmm_4, _shape_param_11);  bmm_4 = _shape_param_11 = None
        permute_10: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_11, [0, 1, 2, 4, 3]);  view_11 = None
        permute_11: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_9, [3, 0, 1, 4, 2]);  permute_9 = None
        squeeze_3: "bf16[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_11, 4);  permute_11 = None
        permute_12: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_10, [2, 0, 1, 4, 3]);  permute_10 = None
        squeeze_4: "bf16[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_12, 4);  permute_12 = None
        convert_element_type_3: "f32[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_4, torch.float32);  squeeze_4 = None
        convert_element_type_4: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.bfloat16)
        sum_2: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(convert_element_type_3, [0, 1], True, dtype = torch.float32);  convert_element_type_3 = None
        view_12: "f32[16, 64]" = torch.ops.aten.view.default(sum_2, _shape_param_12);  sum_2 = _shape_param_12 = None
        add: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(convert_element_type_2, convert_element_type_4);  convert_element_type_2 = convert_element_type_4 = None
        view_13: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.view.default(squeeze_1, _shape_param_13);  squeeze_1 = _shape_param_13 = None
        permute_13: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(view_13, [0, 1, 4, 2, 3]);  view_13 = None
        clone: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
        view_14: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(clone, _shape_param_14);  clone = _shape_param_14 = None
        squeeze_5: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_14, 0);  view_14 = None
        mm_1: "bf16[1024, 1024]" = torch.ops.aten.mm.default(arg11_1, squeeze_5);  arg11_1 = squeeze_5 = None
        unsqueeze: "bf16[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_1, 0);  mm_1 = None
        view_15: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.view.default(unsqueeze, _shape_param_15);  unsqueeze = _shape_param_15 = None
        squeeze_6: "bf16[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(view_15, 4);  view_15 = None
        squeeze_7: "bf16[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_6, 3);  squeeze_6 = None
        convert_element_type_5: "f32[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_7, torch.float32);  squeeze_7 = None
        view_16: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(squeeze, _shape_param_16);  squeeze = _shape_param_16 = None
        permute_14: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.permute.default(view_16, [0, 1, 4, 2, 3]);  view_16 = None
        clone_1: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_17: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_1, _shape_param_17);  clone_1 = _shape_param_17 = None
        squeeze_8: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_17, 0);  view_17 = None
        mm_2: "bf16[1024, 1024]" = torch.ops.aten.mm.default(arg12_1, squeeze_8)
        unsqueeze_1: "bf16[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_2, 0);  mm_2 = None
        mm_3: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_8, arg13_1);  squeeze_8 = arg13_1 = None
        unsqueeze_2: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_3, 0);  mm_3 = None
        view_18: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.view.default(unsqueeze_1, _shape_param_18);  unsqueeze_1 = _shape_param_18 = None
        view_19: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.view.default(unsqueeze_2, _shape_param_19);  unsqueeze_2 = _shape_param_19 = None
        squeeze_9: "bf16[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(view_18, 4);  view_18 = None
        squeeze_10: "bf16[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_9, 3);  squeeze_9 = None
        squeeze_11: "bf16[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(view_19, 4);  view_19 = None
        squeeze_12: "bf16[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze_11, 3);  squeeze_11 = None
        convert_element_type_6: "f32[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_10, torch.float32);  squeeze_10 = None
        convert_element_type_7: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(squeeze_12, torch.float32);  squeeze_12 = None
        add_1: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(arg14_1, convert_element_type_7);  arg14_1 = convert_element_type_7 = None
        view_20: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(squeeze_3, _shape_param_20);  squeeze_3 = _shape_param_20 = None
        permute_15: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.permute.default(view_20, [0, 1, 4, 2, 3]);  view_20 = None
        clone_2: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_21: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_2, _shape_param_21);  clone_2 = _shape_param_21 = None
        squeeze_13: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_21, 0);  view_21 = None
        mm_4: "bf16[1024, 1024]" = torch.ops.aten.mm.default(arg12_1, squeeze_13)
        unsqueeze_3: "bf16[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_4, 0);  mm_4 = None
        mm_5: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_13, arg15_1);  squeeze_13 = arg15_1 = None
        unsqueeze_4: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_5, 0);  mm_5 = None
        view_22: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.view.default(unsqueeze_3, _shape_param_22);  unsqueeze_3 = _shape_param_22 = None
        view_23: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.view.default(unsqueeze_4, _shape_param_23);  unsqueeze_4 = _shape_param_23 = None
        squeeze_14: "bf16[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(view_22, 4);  view_22 = None
        squeeze_15: "bf16[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_14, 3);  squeeze_14 = None
        squeeze_16: "bf16[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(view_23, 4);  view_23 = None
        squeeze_17: "bf16[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze_16, 3);  squeeze_16 = None
        convert_element_type_8: "f32[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_15, torch.float32);  squeeze_15 = None
        convert_element_type_9: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(squeeze_17, torch.float32);  squeeze_17 = None
        add_2: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(add_1, convert_element_type_9);  add_1 = convert_element_type_9 = None
        view_24: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(add, _shape_param_24);  add = _shape_param_24 = None
        permute_16: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.permute.default(view_24, [0, 1, 4, 2, 3]);  view_24 = None
        clone_3: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_25: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_3, _shape_param_25);  clone_3 = _shape_param_25 = None
        squeeze_18: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_25, 0);  view_25 = None
        mm_6: "bf16[1024, 1024]" = torch.ops.aten.mm.default(arg12_1, squeeze_18);  arg12_1 = None
        unsqueeze_5: "bf16[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_6, 0);  mm_6 = None
        mm_7: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_18, arg16_1);  squeeze_18 = arg16_1 = None
        unsqueeze_6: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_7, 0);  mm_7 = None
        view_26: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.view.default(unsqueeze_5, _shape_param_26);  unsqueeze_5 = _shape_param_26 = None
        view_27: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.view.default(unsqueeze_6, _shape_param_27);  unsqueeze_6 = _shape_param_27 = None
        squeeze_19: "bf16[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(view_26, 4);  view_26 = None
        squeeze_20: "bf16[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_19, 3);  squeeze_19 = None
        squeeze_21: "bf16[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(view_27, 4);  view_27 = None
        squeeze_22: "bf16[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze_21, 3);  squeeze_21 = None
        convert_element_type_10: "f32[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_20, torch.float32);  squeeze_20 = None
        convert_element_type_11: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(squeeze_22, torch.float32);  squeeze_22 = None
        add_3: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(add_2, convert_element_type_11);  add_2 = convert_element_type_11 = None
        convert_element_type_12: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.float32);  arg17_1 = None
        mul: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
        mul_1: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_3, mul);  add_3 = mul = None
        ge: "b8[512, 16]" = torch.ops.aten.ge.Scalar(arg18_1, 0)
        lt: "b8[512, 16]" = torch.ops.aten.lt.Scalar(arg18_1, 32000)
        bitwise_and: "b8[512, 16]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[512, 16]" = torch.ops.aten.ne.Scalar(arg18_1, -1)
        bitwise_and_1: "b8[512, 16]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze_7: "b8[512, 16, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full: "f32[32000, 1024]" = torch.ops.aten.full.default(_shape_param_28, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_28 = None
        _unsafe_masked_index_put_accumulate: "f32[32000, 1024]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full, unsqueeze_7, [arg18_1], mul_1);  full = unsqueeze_7 = arg18_1 = mul_1 = None
        add_4: "f32[32000, 1024]" = torch.ops.aten.add.Tensor(convert_element_type, _unsafe_masked_index_put_accumulate);  convert_element_type = _unsafe_masked_index_put_accumulate = None
        return (view_7, view_12, convert_element_type_5, convert_element_type_6, convert_element_type_8, convert_element_type_10, add_4)



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
