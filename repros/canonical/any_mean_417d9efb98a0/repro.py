"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_base_infer_000
Pattern hash: 417d9efb98a0
Shape hash: a25c0aae
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 768], f16), T([1, 2048, 768], f16), T([768], f16), S([1, 2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]), S([2048, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_76: "f16[2048, 768]", convert_element_type_128: "f16[1, 2048, 768]", arg99_1: "f16[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24):
        # No stacktrace found for following nodes
        view_default: "f16[1, 2048, 768]" = torch.ops.aten.view.default(mm_76, _shape_param_0);  mm_76 = _shape_param_0 = None
        add_tensor: "f16[1, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_128, view_default);  convert_element_type_128 = view_default = None
        convert_element_type_default: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32)
        isinf_default: "b8[1, 2048, 768]" = torch.ops.aten.isinf.default(add_tensor);  add_tensor = None
        any_default: "b8[]" = torch.ops.aten.any.default(isinf_default);  isinf_default = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[]" = torch.ops.aten.where.self(any_default, full_default, full_default_1);  any_default = full_default = full_default_1 = None
        neg_default: "f32[]" = torch.ops.aten.neg.default(where_self)
        clamp_min_tensor: "f32[1, 2048, 768]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_default, neg_default);  convert_element_type_default = neg_default = None
        clamp_max_tensor: "f32[1, 2048, 768]" = torch.ops.aten.clamp_max.Tensor(clamp_min_tensor, where_self);  clamp_min_tensor = where_self = None
        convert_element_type_default_1: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(clamp_max_tensor, torch.float16);  clamp_max_tensor = None
        convert_element_type_default_2: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_1, torch.float32)
        pow_tensor_scalar: "f32[1, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 2);  convert_element_type_default_2 = None
        mean_dim: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor_1: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, rsqrt_default);  convert_element_type_default_1 = rsqrt_default = None
        convert_element_type_default_3: "f16[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float16);  mul_tensor = None
        mul_tensor_1: "f16[1, 2048, 768]" = torch.ops.aten.mul.Tensor(arg99_1, convert_element_type_default_3);  arg99_1 = convert_element_type_default_3 = None
        view_default_1: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_1);  _shape_param_1 = None
        view_default_2: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_2);  _shape_param_2 = None
        view_default_3: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_3);  _shape_param_3 = None
        view_default_4: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_4);  _shape_param_4 = None
        view_default_5: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_5);  _shape_param_5 = None
        view_default_6: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_6);  _shape_param_6 = None
        view_default_7: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_7);  _shape_param_7 = None
        view_default_8: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_8);  _shape_param_8 = None
        view_default_9: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_9);  _shape_param_9 = None
        view_default_10: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_10);  _shape_param_10 = None
        view_default_11: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_11);  _shape_param_11 = None
        view_default_12: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_12);  _shape_param_12 = None
        view_default_13: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_13);  _shape_param_13 = None
        view_default_14: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_14);  _shape_param_14 = None
        view_default_15: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_15);  _shape_param_15 = None
        view_default_16: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_16);  _shape_param_16 = None
        view_default_17: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_17);  _shape_param_17 = None
        view_default_18: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_18);  _shape_param_18 = None
        view_default_19: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_19);  _shape_param_19 = None
        view_default_20: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_20);  _shape_param_20 = None
        view_default_21: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_21);  _shape_param_21 = None
        view_default_22: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_22);  _shape_param_22 = None
        view_default_23: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_23);  _shape_param_23 = None
        view_default_24: "f16[2048, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_24);  mul_tensor_1 = _shape_param_24 = None
        return (view_default_1, view_default_2, view_default_3, view_default_4, view_default_5, view_default_6, view_default_7, view_default_8, view_default_9, view_default_10, view_default_11, view_default_12, view_default_13, view_default_14, view_default_15, view_default_16, view_default_17, view_default_18, view_default_19, view_default_20, view_default_21, view_default_22, view_default_23, view_default_24)



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
