"""
Standalone repro captured via capture_hook.
Label: torchbench_functorch_dp_cifar10_train
Pattern hash: 81f672a568a0
Shape hash: 05c56638
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
    def forward(self, arg0_1: "bf16[128, 64, 8, 8]", arg1_1: "f32[128, 64, 8, 8]", arg2_1: "i8[128, 64, 8, 8]", arg3_1: "bf16[128, 64, 16, 16]", arg4_1: "f32[128, 32, 1, 1]", arg5_1: "f32[128, 32, 1, 1]", arg6_1: "f32[64]", arg7_1: "f32[64]", arg8_1: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 64, 8, 8]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        add: "f32[128, 64, 8, 8]" = torch.ops.aten.add.Tensor(arg1_1, convert_element_type);  arg1_1 = convert_element_type = None
        full: "f32[8192, 256]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        clone: "f32[128, 64, 8, 8]" = torch.ops.aten.clone.default(add, memory_format = torch.contiguous_format);  add = None
        _unsafe_view: "f32[8192, 64]" = torch.ops.aten._unsafe_view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        _low_memory_max_pool_offsets_to_indices: "i64[128, 64, 8, 8]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(arg2_1, _shape_param_2, _shape_param_3, _shape_param_4, [1, 1], [1, 1]);  arg2_1 = _shape_param_2 = _shape_param_3 = _shape_param_4 = None
        clone_1: "i64[128, 64, 8, 8]" = torch.ops.aten.clone.default(_low_memory_max_pool_offsets_to_indices, memory_format = torch.contiguous_format);  _low_memory_max_pool_offsets_to_indices = None
        _unsafe_view_1: "i64[8192, 64]" = torch.ops.aten._unsafe_view.default(clone_1, _shape_param_5);  clone_1 = _shape_param_5 = None
        scatter_add: "f32[8192, 256]" = torch.ops.aten.scatter_add.default(full, 1, _unsafe_view_1, _unsafe_view);  full = _unsafe_view_1 = _unsafe_view = None
        view: "f32[128, 64, 16, 16]" = torch.ops.aten.view.default(scatter_add, _shape_param_6);  scatter_add = _shape_param_6 = None
        convert_element_type_1: "f32[128, 64, 16, 16]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        view_1: "f32[128, 32, 2, 256]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_7);  _shape_param_7 = None
        sub: "f32[128, 32, 2, 256]" = torch.ops.aten.sub.Tensor(view_1, arg4_1)
        mul: "f32[128, 32, 2, 256]" = torch.ops.aten.mul.Tensor(sub, arg5_1);  sub = None
        view_2: "f32[128, 64, 16, 16]" = torch.ops.aten.view.default(mul, _shape_param_8);  mul = _shape_param_8 = None
        unsqueeze: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_1: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2)
        unsqueeze_2: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_1: "f32[128, 64, 16, 16]" = torch.ops.aten.mul.Tensor(view_2, unsqueeze_2);  view_2 = unsqueeze_2 = None
        unsqueeze_3: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(arg7_1, 0);  arg7_1 = None
        unsqueeze_4: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        add_1: "f32[128, 64, 16, 16]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        relu: "f32[128, 64, 16, 16]" = torch.ops.aten.relu.default(add_1);  add_1 = None
        le: "b8[128, 64, 16, 16]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where: "f32[128, 64, 16, 16]" = torch.ops.aten.where.self(le, arg8_1, view);  le = arg8_1 = view = None
        mul_2: "f32[128, 64, 16, 16]" = torch.ops.aten.mul.Tensor(where, convert_element_type_1);  convert_element_type_1 = None
        view_3: "f32[128, 64, 256]" = torch.ops.aten.view.default(mul_2, _shape_param_9);  mul_2 = _shape_param_9 = None
        sum_1: "f32[128, 64]" = torch.ops.aten.sum.dim_IntList(view_3, [2]);  view_3 = None
        view_4: "f32[128, 64, 256]" = torch.ops.aten.view.default(where, _shape_param_10);  _shape_param_10 = None
        sum_2: "f32[128, 64]" = torch.ops.aten.sum.dim_IntList(view_4, [2]);  view_4 = None
        mul_3: "f32[128, 64]" = torch.ops.aten.mul.Tensor(sum_1, unsqueeze)
        view_5: "f32[128, 32, 2]" = torch.ops.aten.view.default(mul_3, _shape_param_11);  mul_3 = _shape_param_11 = None
        sum_3: "f32[128, 32]" = torch.ops.aten.sum.dim_IntList(view_5, [2]);  view_5 = None
        mul_4: "f32[128, 64]" = torch.ops.aten.mul.Tensor(sum_2, unsqueeze);  unsqueeze = None
        view_6: "f32[128, 32, 2]" = torch.ops.aten.view.default(mul_4, _shape_param_12);  mul_4 = _shape_param_12 = None
        sum_4: "f32[128, 32]" = torch.ops.aten.sum.dim_IntList(view_6, [2]);  view_6 = None
        squeeze: "f32[128, 32]" = torch.ops.aten.squeeze.dims(arg5_1, [2, 3]);  arg5_1 = None
        unsqueeze_6: "f32[128, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze, -1)
        view_7: "f32[1, 32, 2]" = torch.ops.aten.view.default(arg6_1, _shape_param_13);  arg6_1 = _shape_param_13 = None
        mul_5: "f32[128, 32, 2]" = torch.ops.aten.mul.Tensor(unsqueeze_6, view_7);  view_7 = None
        squeeze_1: "f32[128, 32]" = torch.ops.aten.squeeze.dims(arg4_1, [2, 3]);  arg4_1 = None
        mul_6: "f32[128, 32]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_1)
        sub_1: "f32[128, 32]" = torch.ops.aten.sub.Tensor(mul_6, sum_3);  mul_6 = sum_3 = None
        mul_7: "f32[128, 32]" = torch.ops.aten.mul.Tensor(sub_1, squeeze);  sub_1 = None
        mul_8: "f32[128, 32]" = torch.ops.aten.mul.Tensor(mul_7, squeeze);  mul_7 = None
        mul_9: "f32[128, 32]" = torch.ops.aten.mul.Tensor(mul_8, squeeze);  mul_8 = None
        mul_10: "f32[128, 32]" = torch.ops.aten.mul.Tensor(mul_9, 0.001953125);  mul_9 = None
        neg: "f32[128, 32]" = torch.ops.aten.neg.default(mul_10)
        mul_11: "f32[128, 32]" = torch.ops.aten.mul.Tensor(neg, squeeze_1);  neg = None
        mul_12: "f32[128, 32]" = torch.ops.aten.mul.Tensor(sum_4, squeeze);  sum_4 = squeeze = None
        mul_13: "f32[128, 32]" = torch.ops.aten.mul.Tensor(mul_12, 0.001953125);  mul_12 = None
        sub_2: "f32[128, 32]" = torch.ops.aten.sub.Tensor(mul_11, mul_13);  mul_11 = mul_13 = None
        unsqueeze_7: "f32[128, 32, 2, 1]" = torch.ops.aten.unsqueeze.default(mul_5, -1);  mul_5 = None
        unsqueeze_8: "f32[128, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_10, -1);  mul_10 = None
        unsqueeze_9: "f32[128, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        unsqueeze_10: "f32[128, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_2, -1);  sub_2 = None
        unsqueeze_11: "f32[128, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        view_8: "f32[128, 32, 2, 256]" = torch.ops.aten.view.default(where, _shape_param_14);  where = _shape_param_14 = None
        mul_14: "f32[128, 32, 2, 256]" = torch.ops.aten.mul.Tensor(view_8, unsqueeze_7);  view_8 = unsqueeze_7 = None
        mul_15: "f32[128, 32, 2, 256]" = torch.ops.aten.mul.Tensor(view_1, unsqueeze_9);  view_1 = unsqueeze_9 = None
        add_2: "f32[128, 32, 2, 256]" = torch.ops.aten.add.Tensor(mul_14, mul_15);  mul_14 = mul_15 = None
        add_3: "f32[128, 32, 2, 256]" = torch.ops.aten.add.Tensor(add_2, unsqueeze_11);  add_2 = unsqueeze_11 = None
        view_9: "f32[128, 64, 16, 16]" = torch.ops.aten.view.default(add_3, _shape_param_15);  add_3 = _shape_param_15 = None
        view_10: "f32[128, 32, 2]" = torch.ops.aten.view.default(sum_1, _shape_param_16);  sum_1 = _shape_param_16 = None
        view_11: "f32[128, 32, 2]" = torch.ops.aten.view.default(sum_2, _shape_param_17);  _shape_param_17 = None
        unsqueeze_12: "f32[128, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_1, -1);  squeeze_1 = None
        mul_16: "f32[128, 32, 2]" = torch.ops.aten.mul.Tensor(view_11, unsqueeze_12);  view_11 = unsqueeze_12 = None
        sub_3: "f32[128, 32, 2]" = torch.ops.aten.sub.Tensor(view_10, mul_16);  view_10 = mul_16 = None
        mul_17: "f32[128, 32, 2]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_6);  sub_3 = unsqueeze_6 = None
        sum_5: "f32[32, 2]" = torch.ops.aten.sum.dim_IntList(mul_17, [0]);  mul_17 = None
        view_12: "f32[64]" = torch.ops.aten.view.default(sum_5, _shape_param_18);  sum_5 = _shape_param_18 = None
        sum_6: "f32[64]" = torch.ops.aten.sum.dim_IntList(sum_2, [0]);  sum_2 = None
        convert_element_type_2: "bf16[128, 64, 16, 16]" = torch.ops.prims.convert_element_type.default(view_9, torch.bfloat16);  view_9 = None
        return (view_12, sum_6, convert_element_type_2)



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
