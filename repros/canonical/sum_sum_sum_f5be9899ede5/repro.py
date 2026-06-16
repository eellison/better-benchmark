"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_unet_train
Pattern hash: f5be9899ede5
Shape hash: b518aad7
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
    def forward(self, arg0_1: "bf16[1, 128, 640, 959]", arg1_1: "f32[640, 1]", arg2_1: "f32[958]", arg3_1: "i64[640, 1]", arg4_1: "i64[958]", arg5_1: "i64[958]", arg6_1: "i64[640, 1]", arg7_1: "bf16[1, 64, 320, 479]", arg8_1: "f32[1, 64, 1, 1]", arg9_1: "f32[1, 64, 1, 1]", arg10_1: "f32[64]", arg11_1: "f32[64]", arg12_1: "bf16[]", _shape_param_0):
        # No stacktrace found for following nodes
        slice_1: "bf16[1, 64, 640, 959]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 64, 128);  arg0_1 = None
        constant_pad_nd: "bf16[1, 64, 640, 958]" = torch.ops.aten.constant_pad_nd.default(slice_1, [0, -1, 0, 0]);  slice_1 = None
        convert_element_type: "f32[1, 64, 640, 958]" = torch.ops.prims.convert_element_type.default(constant_pad_nd, torch.float32);  constant_pad_nd = None
        mul: "f32[1, 64, 640, 958]" = torch.ops.aten.mul.Tensor(convert_element_type, arg1_1);  arg1_1 = None
        neg: "f32[1, 64, 640, 958]" = torch.ops.aten.neg.default(mul)
        add: "f32[1, 64, 640, 958]" = torch.ops.aten.add.Tensor(convert_element_type, neg);  convert_element_type = neg = None
        mul_1: "f32[1, 64, 640, 958]" = torch.ops.aten.mul.Tensor(mul, arg2_1)
        neg_1: "f32[1, 64, 640, 958]" = torch.ops.aten.neg.default(mul_1)
        add_1: "f32[1, 64, 640, 958]" = torch.ops.aten.add.Tensor(mul, neg_1);  mul = neg_1 = None
        mul_2: "f32[1, 64, 640, 958]" = torch.ops.aten.mul.Tensor(add, arg2_1);  arg2_1 = None
        neg_2: "f32[1, 64, 640, 958]" = torch.ops.aten.neg.default(mul_2)
        add_2: "f32[1, 64, 640, 958]" = torch.ops.aten.add.Tensor(add, neg_2);  add = neg_2 = None
        full: "f32[1, 64, 320, 479]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        index_put: "f32[1, 64, 320, 479]" = torch.ops.aten.index_put.default(full, [None, None, arg3_1, arg4_1], mul_1, True);  mul_1 = None
        index_put_1: "f32[1, 64, 320, 479]" = torch.ops.aten.index_put.default(full, [None, None, arg3_1, arg5_1], add_1, True);  arg3_1 = add_1 = None
        add_3: "f32[1, 64, 320, 479]" = torch.ops.aten.add.Tensor(index_put, index_put_1);  index_put = index_put_1 = None
        index_put_2: "f32[1, 64, 320, 479]" = torch.ops.aten.index_put.default(full, [None, None, arg6_1, arg4_1], mul_2, True);  arg4_1 = mul_2 = None
        add_4: "f32[1, 64, 320, 479]" = torch.ops.aten.add.Tensor(add_3, index_put_2);  add_3 = index_put_2 = None
        index_put_3: "f32[1, 64, 320, 479]" = torch.ops.aten.index_put.default(full, [None, None, arg6_1, arg5_1], add_2, True);  full = arg6_1 = arg5_1 = add_2 = None
        add_5: "f32[1, 64, 320, 479]" = torch.ops.aten.add.Tensor(add_4, index_put_3);  add_4 = index_put_3 = None
        convert_element_type_1: "bf16[1, 64, 320, 479]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None
        sub: "f32[1, 64, 320, 479]" = torch.ops.aten.sub.Tensor(arg7_1, arg8_1)
        mul_3: "f32[1, 64, 320, 479]" = torch.ops.aten.mul.Tensor(sub, arg9_1);  sub = None
        unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1)
        unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_4: "f32[1, 64, 320, 479]" = torch.ops.aten.mul.Tensor(mul_3, unsqueeze_1);  mul_3 = unsqueeze_1 = None
        unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
        unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_6: "f32[1, 64, 320, 479]" = torch.ops.aten.add.Tensor(mul_4, unsqueeze_3);  mul_4 = unsqueeze_3 = None
        convert_element_type_2: "bf16[1, 64, 320, 479]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        relu: "bf16[1, 64, 320, 479]" = torch.ops.aten.relu.default(convert_element_type_2);  convert_element_type_2 = None
        le: "b8[1, 64, 320, 479]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where: "bf16[1, 64, 320, 479]" = torch.ops.aten.where.self(le, arg12_1, convert_element_type_1);  le = arg12_1 = convert_element_type_1 = None
        convert_element_type_3: "f32[1, 64, 320, 479]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze: "f32[64]" = torch.ops.aten.squeeze.dims(arg8_1, [0, 2, 3]);  arg8_1 = None
        unsqueeze_4: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_5: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        unsqueeze_6: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None
        sum_1: "f32[64]" = torch.ops.aten.sum.dim_IntList(convert_element_type_3, [0, 2, 3])
        convert_element_type_4: "f32[1, 64, 320, 479]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.float32);  arg7_1 = None
        sub_1: "f32[1, 64, 320, 479]" = torch.ops.aten.sub.Tensor(convert_element_type_4, unsqueeze_6);  convert_element_type_4 = unsqueeze_6 = None
        mul_5: "f32[1, 64, 320, 479]" = torch.ops.aten.mul.Tensor(convert_element_type_3, sub_1)
        sum_2: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 2, 3]);  mul_5 = None
        mul_6: "f32[64]" = torch.ops.aten.mul.Tensor(sum_1, 6.524008350730689e-06)
        unsqueeze_7: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_6, 0);  mul_6 = None
        unsqueeze_8: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None
        unsqueeze_9: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 3);  unsqueeze_8 = None
        mul_7: "f32[64]" = torch.ops.aten.mul.Tensor(sum_2, 6.524008350730689e-06)
        squeeze_1: "f32[64]" = torch.ops.aten.squeeze.dims(arg9_1, [0, 2, 3]);  arg9_1 = None
        mul_8: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(mul_7, mul_8);  mul_7 = mul_8 = None
        unsqueeze_10: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_9, 0);  mul_9 = None
        unsqueeze_11: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        unsqueeze_12: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 3);  unsqueeze_11 = None
        mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, arg10_1);  arg10_1 = None
        unsqueeze_13: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_10, 0);  mul_10 = None
        unsqueeze_14: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 2);  unsqueeze_13 = None
        unsqueeze_15: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 3);  unsqueeze_14 = None
        mul_11: "f32[1, 64, 320, 479]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_12);  sub_1 = unsqueeze_12 = None
        sub_2: "f32[1, 64, 320, 479]" = torch.ops.aten.sub.Tensor(convert_element_type_3, mul_11);  convert_element_type_3 = mul_11 = None
        sub_3: "f32[1, 64, 320, 479]" = torch.ops.aten.sub.Tensor(sub_2, unsqueeze_9);  sub_2 = unsqueeze_9 = None
        mul_12: "f32[1, 64, 320, 479]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_15);  sub_3 = unsqueeze_15 = None
        mul_13: "f32[64]" = torch.ops.aten.mul.Tensor(sum_2, squeeze_1);  sum_2 = squeeze_1 = None
        convert_element_type_5: "bf16[1, 64, 320, 479]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        sum_3: "bf16[64]" = torch.ops.aten.sum.dim_IntList(convert_element_type_5, [0, 2, 3])
        convert_element_type_6: "f32[64]" = torch.ops.prims.convert_element_type.default(sum_3, torch.float32);  sum_3 = None
        return (sum_1, mul_13, convert_element_type_5, convert_element_type_6)



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
