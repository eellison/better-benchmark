"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf_cuda_h100-6-9-linux.aws.h100_graph66
Pattern hash: b00280229d23
Shape hash: 7f26fa30
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([], f32), T([1], f32), T([1], f32), T([1], f32), T([1], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg3_1: "f32[]", arg4_1: "f32[1]", arg1_1: "f32[1]", arg2_1: "f32[1]", arg0_1: "f32[1]"):
        # No stacktrace found for following nodes
        _foreach_add_scalar = torch.ops.aten._foreach_add.Scalar([arg3_1], 1)
        getitem: "f32[]" = _foreach_add_scalar[0];  _foreach_add_scalar = None
        _foreach_sub_list = torch.ops.aten._foreach_sub.List([arg4_1], [arg1_1])
        getitem_1: "f32[1]" = _foreach_sub_list[0];  _foreach_sub_list = None
        full_default: "f32[1]" = torch.ops.aten.full.default([1], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _foreach_addcmul_scalar = torch.ops.aten._foreach_addcmul.Scalar([arg1_1], [full_default], [getitem_1]);  full_default = getitem_1 = None
        getitem_2: "f32[1]" = _foreach_addcmul_scalar[0];  _foreach_addcmul_scalar = None
        _foreach_mul_scalar = torch.ops.aten._foreach_mul.Scalar([arg2_1], 0.999)
        getitem_3: "f32[1]" = _foreach_mul_scalar[0];  _foreach_mul_scalar = None
        _foreach_addcmul_scalar_1 = torch.ops.aten._foreach_addcmul.Scalar([getitem_3], [arg4_1], [arg4_1], 0.0010000000000000009);  getitem_3 = arg4_1 = None
        getitem_4: "f32[1]" = _foreach_addcmul_scalar_1[0];  _foreach_addcmul_scalar_1 = None
        _foreach_pow_scalar_and_tensor = torch.ops.aten._foreach_pow.ScalarAndTensor(0.9, [getitem])
        getitem_5: "f32[]" = _foreach_pow_scalar_and_tensor[0];  _foreach_pow_scalar_and_tensor = None
        _foreach_pow_scalar_and_tensor_1 = torch.ops.aten._foreach_pow.ScalarAndTensor(0.999, [getitem])
        getitem_6: "f32[]" = _foreach_pow_scalar_and_tensor_1[0];  _foreach_pow_scalar_and_tensor_1 = None
        _foreach_sub_scalar = torch.ops.aten._foreach_sub.Scalar([getitem_5], 1);  getitem_5 = None
        getitem_7: "f32[]" = _foreach_sub_scalar[0];  _foreach_sub_scalar = None
        _foreach_sub_scalar_1 = torch.ops.aten._foreach_sub.Scalar([getitem_6], 1);  getitem_6 = None
        getitem_8: "f32[]" = _foreach_sub_scalar_1[0];  _foreach_sub_scalar_1 = None
        _foreach_neg_default = torch.ops.aten._foreach_neg.default([getitem_8]);  getitem_8 = None
        getitem_9: "f32[]" = _foreach_neg_default[0];  _foreach_neg_default = None
        _foreach_div_scalar = torch.ops.aten._foreach_div.Scalar([getitem_7], 0.01);  getitem_7 = None
        getitem_10: "f32[]" = _foreach_div_scalar[0];  _foreach_div_scalar = None
        _foreach_reciprocal_default = torch.ops.aten._foreach_reciprocal.default([getitem_10]);  getitem_10 = None
        getitem_11: "f32[]" = _foreach_reciprocal_default[0];  _foreach_reciprocal_default = None
        _foreach_sqrt_default = torch.ops.aten._foreach_sqrt.default([getitem_9]);  getitem_9 = None
        getitem_12: "f32[]" = _foreach_sqrt_default[0];  _foreach_sqrt_default = None
        _foreach_sqrt_default_1 = torch.ops.aten._foreach_sqrt.default([getitem_4])
        getitem_13: "f32[1]" = _foreach_sqrt_default_1[0];  _foreach_sqrt_default_1 = None
        _foreach_div_list = torch.ops.aten._foreach_div.List([getitem_13], [getitem_12]);  getitem_13 = getitem_12 = None
        getitem_14: "f32[1]" = _foreach_div_list[0];  _foreach_div_list = None
        _foreach_add_scalar_1 = torch.ops.aten._foreach_add.Scalar([getitem_14], 1e-08);  getitem_14 = None
        getitem_15: "f32[1]" = _foreach_add_scalar_1[0];  _foreach_add_scalar_1 = None
        _foreach_div_list_1 = torch.ops.aten._foreach_div.List([getitem_15], [getitem_11]);  getitem_15 = getitem_11 = None
        getitem_16: "f32[1]" = _foreach_div_list_1[0];  _foreach_div_list_1 = None
        _foreach_addcdiv_scalar = torch.ops.aten._foreach_addcdiv.Scalar([arg0_1], [getitem_2], [getitem_16]);  getitem_16 = None
        getitem_17: "f32[1]" = _foreach_addcdiv_scalar[0];  _foreach_addcdiv_scalar = None
        copy__default: "f32[1]" = torch.ops.aten.copy_.default(arg0_1, getitem_17);  arg0_1 = getitem_17 = None
        copy__default_1: "f32[1]" = torch.ops.aten.copy_.default(arg1_1, getitem_2);  arg1_1 = getitem_2 = None
        copy__default_2: "f32[1]" = torch.ops.aten.copy_.default(arg2_1, getitem_4);  arg2_1 = getitem_4 = None
        copy__default_3: "f32[]" = torch.ops.aten.copy_.default(arg3_1, getitem);  arg3_1 = getitem = None
        return (copy__default, copy__default_1, copy__default_2, copy__default_3)


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
