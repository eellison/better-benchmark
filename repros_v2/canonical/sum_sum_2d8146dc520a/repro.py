"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train
Pattern hash: 2d8146dc520a
Shape hash: e5f2e3e0
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
    def forward(self, arg0_1: "bf16[128, 192, 35, 35]", arg1_1: "bf16[128, 192, 35, 35]", arg2_1: "bf16[128, 192, 35, 35]", arg3_1: "bf16[128, 192, 35, 35]", arg4_1: "bf16[128, 192, 35, 35]", arg5_1: "i8[128, 192, 35, 35]", arg6_1: "bf16[128, 192, 71, 71]", arg7_1: "f32[1, 192, 1, 1]", arg8_1: "f32[1, 192, 1, 1]", arg9_1: "f32[192]", arg10_1: "f32[192]", arg11_1: "bf16[]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        avg_pool2d_backward: "bf16[128, 192, 35, 35]" = torch.ops.aten.avg_pool2d_backward.default(arg0_1, arg1_1, [3, 3], [1, 1], [1, 1], False, True, None);  arg0_1 = arg1_1 = None
        add: "bf16[128, 192, 35, 35]" = torch.ops.aten.add.Tensor(avg_pool2d_backward, arg2_1);  avg_pool2d_backward = arg2_1 = None
        add_1: "bf16[128, 192, 35, 35]" = torch.ops.aten.add.Tensor(add, arg3_1);  add = arg3_1 = None
        add_2: "bf16[128, 192, 35, 35]" = torch.ops.aten.add.Tensor(add_1, arg4_1);  add_1 = arg4_1 = None
        full: "f32[24576, 5041]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        clone: "bf16[128, 192, 35, 35]" = torch.ops.aten.clone.default(add_2, memory_format = torch.contiguous_format);  add_2 = None
        view: "bf16[24576, 1225]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        _low_memory_max_pool_offsets_to_indices: "i64[128, 192, 35, 35]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(arg5_1, _shape_param_2, _shape_param_3, _shape_param_4, [0, 0], [1, 1]);  arg5_1 = _shape_param_2 = _shape_param_3 = _shape_param_4 = None
        clone_1: "i64[128, 192, 35, 35]" = torch.ops.aten.clone.default(_low_memory_max_pool_offsets_to_indices, memory_format = torch.contiguous_format);  _low_memory_max_pool_offsets_to_indices = None
        view_1: "i64[24576, 1225]" = torch.ops.aten.view.default(clone_1, _shape_param_5);  clone_1 = _shape_param_5 = None
        convert_element_type: "f32[24576, 1225]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        scatter_add: "f32[24576, 5041]" = torch.ops.aten.scatter_add.default(full, 1, view_1, convert_element_type);  full = view_1 = convert_element_type = None
        view_2: "f32[128, 192, 71, 71]" = torch.ops.aten.view.default(scatter_add, _shape_param_6);  scatter_add = _shape_param_6 = None
        convert_element_type_1: "bf16[128, 192, 71, 71]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        clone_2: "bf16[128, 192, 71, 71]" = torch.ops.aten.clone.default(convert_element_type_1, memory_format = torch.channels_last);  convert_element_type_1 = None
        sub: "f32[128, 192, 71, 71]" = torch.ops.aten.sub.Tensor(arg6_1, arg7_1)
        mul: "f32[128, 192, 71, 71]" = torch.ops.aten.mul.Tensor(sub, arg8_1);  sub = None
        unsqueeze: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1)
        unsqueeze_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_1: "f32[128, 192, 71, 71]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_3: "f32[128, 192, 71, 71]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type_2: "bf16[128, 192, 71, 71]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        relu: "bf16[128, 192, 71, 71]" = torch.ops.aten.relu.default(convert_element_type_2);  convert_element_type_2 = None
        le: "b8[128, 192, 71, 71]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where: "bf16[128, 192, 71, 71]" = torch.ops.aten.where.self(le, arg11_1, clone_2);  le = arg11_1 = clone_2 = None
        convert_element_type_3: "f32[128, 192, 71, 71]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze: "f32[192]" = torch.ops.aten.squeeze.dims(arg7_1, [0, 2, 3]);  arg7_1 = None
        unsqueeze_4: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_5: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        unsqueeze_6: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None
        sum_1: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_3, [0, 2, 3])
        convert_element_type_4: "f32[128, 192, 71, 71]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float32);  arg6_1 = None
        sub_1: "f32[128, 192, 71, 71]" = torch.ops.aten.sub.Tensor(convert_element_type_4, unsqueeze_6);  convert_element_type_4 = unsqueeze_6 = None
        mul_2: "f32[128, 192, 71, 71]" = torch.ops.aten.mul.Tensor(convert_element_type_3, sub_1)
        sum_2: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_2, [0, 2, 3]);  mul_2 = None
        mul_3: "f32[192]" = torch.ops.aten.mul.Tensor(sum_1, 1.5497917079944455e-06)
        unsqueeze_7: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_3, 0);  mul_3 = None
        unsqueeze_8: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None
        unsqueeze_9: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 3);  unsqueeze_8 = None
        mul_4: "f32[192]" = torch.ops.aten.mul.Tensor(sum_2, 1.5497917079944455e-06)
        squeeze_1: "f32[192]" = torch.ops.aten.squeeze.dims(arg8_1, [0, 2, 3]);  arg8_1 = None
        mul_5: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_6: "f32[192]" = torch.ops.aten.mul.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze_10: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_6, 0);  mul_6 = None
        unsqueeze_11: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        unsqueeze_12: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 3);  unsqueeze_11 = None
        mul_7: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_1, arg9_1);  arg9_1 = None
        unsqueeze_13: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_7, 0);  mul_7 = None
        unsqueeze_14: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 2);  unsqueeze_13 = None
        unsqueeze_15: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 3);  unsqueeze_14 = None
        mul_8: "f32[128, 192, 71, 71]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_12);  sub_1 = unsqueeze_12 = None
        sub_2: "f32[128, 192, 71, 71]" = torch.ops.aten.sub.Tensor(convert_element_type_3, mul_8);  convert_element_type_3 = mul_8 = None
        sub_3: "f32[128, 192, 71, 71]" = torch.ops.aten.sub.Tensor(sub_2, unsqueeze_9);  sub_2 = unsqueeze_9 = None
        mul_9: "f32[128, 192, 71, 71]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_15);  sub_3 = unsqueeze_15 = None
        mul_10: "f32[192]" = torch.ops.aten.mul.Tensor(sum_2, squeeze_1);  sum_2 = squeeze_1 = None
        convert_element_type_5: "bf16[128, 192, 71, 71]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        return (sum_1, mul_10, convert_element_type_5)



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
