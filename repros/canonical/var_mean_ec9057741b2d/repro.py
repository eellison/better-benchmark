"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_unet_train
Pattern hash: ec9057741b2d
Shape hash: d2d98efc
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
    def forward(self, arg0_1: "bf16[1, 128, 160, 239]", arg1_1: "f32[128]", arg2_1: "f32[128]", arg3_1: "f32[128]", arg4_1: "f32[128]", arg5_1: "bf16[1, 128, 320, 479]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "f32[1, 128, 160, 239]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type = None
        getitem: "f32[1, 128, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 128, 1, 1]" = var_mean[1];  var_mean = None
        add: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[1, 128, 160, 239]" = torch.ops.aten.sub.Tensor(arg0_1, getitem_1);  arg0_1 = None
        mul: "f32[1, 128, 160, 239]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul_1: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
        mul_2: "f32[128]" = torch.ops.aten.mul.Tensor(arg1_1, 0.9)
        add_1: "f32[128]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_1: "f32[128]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_1, 1.0000261513114883);  squeeze_1 = None
        mul_4: "f32[128]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[128]" = torch.ops.aten.mul.Tensor(arg2_1, 0.9)
        add_2: "f32[128]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_1: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[1, 128, 160, 239]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_3: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_3: "f32[1, 128, 160, 239]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_1: "bf16[1, 128, 160, 239]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        relu: "bf16[1, 128, 160, 239]" = torch.ops.aten.relu.default(convert_element_type_1);  convert_element_type_1 = None
        convert_element_type_2: "f32[1, 128, 160, 239]" = torch.ops.prims.convert_element_type.default(relu, torch.float32);  relu = None
        iota: "i64[320]" = torch.ops.prims.iota.default(320, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_3: "f32[320]" = torch.ops.prims.convert_element_type.default(iota, torch.float32);  iota = None
        mul_7: "f32[320]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.49843260188087773);  convert_element_type_3 = None
        clamp_min: "f32[320]" = torch.ops.aten.clamp_min.default(mul_7, 0.0);  mul_7 = None
        view: "f32[320, 1]" = torch.ops.aten.view.default(clamp_min, _shape_param_0);  clamp_min = _shape_param_0 = None
        convert_element_type_4: "i64[320, 1]" = torch.ops.prims.convert_element_type.default(view, torch.int64)
        add_4: "i64[320, 1]" = torch.ops.aten.add.Tensor(convert_element_type_4, 1)
        clamp_max: "i64[320, 1]" = torch.ops.aten.clamp_max.default(add_4, 159);  add_4 = None
        iota_1: "i64[478]" = torch.ops.prims.iota.default(478, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_5: "f32[478]" = torch.ops.prims.convert_element_type.default(iota_1, torch.float32);  iota_1 = None
        mul_8: "f32[478]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 0.4989517819706499);  convert_element_type_5 = None
        clamp_min_1: "f32[478]" = torch.ops.aten.clamp_min.default(mul_8, 0.0);  mul_8 = None
        convert_element_type_6: "i64[478]" = torch.ops.prims.convert_element_type.default(clamp_min_1, torch.int64)
        add_5: "i64[478]" = torch.ops.aten.add.Tensor(convert_element_type_6, 1)
        clamp_max_1: "i64[478]" = torch.ops.aten.clamp_max.default(add_5, 238);  add_5 = None
        _unsafe_index: "f32[1, 128, 320, 478]" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_2, [None, None, convert_element_type_4, convert_element_type_6])
        _unsafe_index_1: "f32[1, 128, 320, 478]" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_2, [None, None, convert_element_type_4, clamp_max_1])
        _unsafe_index_2: "f32[1, 128, 320, 478]" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_2, [None, None, clamp_max, convert_element_type_6])
        _unsafe_index_3: "f32[1, 128, 320, 478]" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_2, [None, None, clamp_max, clamp_max_1]);  convert_element_type_2 = None
        sub_1: "f32[478]" = torch.ops.aten.sub.Tensor(clamp_min_1, convert_element_type_6);  clamp_min_1 = None
        clamp_min_2: "f32[478]" = torch.ops.aten.clamp_min.default(sub_1, 0.0);  sub_1 = None
        clamp_max_2: "f32[478]" = torch.ops.aten.clamp_max.default(clamp_min_2, 1.0);  clamp_min_2 = None
        sub_2: "f32[1, 128, 320, 478]" = torch.ops.aten.sub.Tensor(_unsafe_index_1, _unsafe_index);  _unsafe_index_1 = None
        mul_9: "f32[1, 128, 320, 478]" = torch.ops.aten.mul.Tensor(sub_2, clamp_max_2);  sub_2 = None
        add_6: "f32[1, 128, 320, 478]" = torch.ops.aten.add.Tensor(_unsafe_index, mul_9);  _unsafe_index = mul_9 = None
        sub_3: "f32[1, 128, 320, 478]" = torch.ops.aten.sub.Tensor(_unsafe_index_3, _unsafe_index_2);  _unsafe_index_3 = None
        mul_10: "f32[1, 128, 320, 478]" = torch.ops.aten.mul.Tensor(sub_3, clamp_max_2);  sub_3 = None
        add_7: "f32[1, 128, 320, 478]" = torch.ops.aten.add.Tensor(_unsafe_index_2, mul_10);  _unsafe_index_2 = mul_10 = None
        sub_4: "f32[320, 1]" = torch.ops.aten.sub.Tensor(view, convert_element_type_4);  view = None
        clamp_min_3: "f32[320, 1]" = torch.ops.aten.clamp_min.default(sub_4, 0.0);  sub_4 = None
        clamp_max_3: "f32[320, 1]" = torch.ops.aten.clamp_max.default(clamp_min_3, 1.0);  clamp_min_3 = None
        sub_5: "f32[1, 128, 320, 478]" = torch.ops.aten.sub.Tensor(add_7, add_6);  add_7 = None
        mul_11: "f32[1, 128, 320, 478]" = torch.ops.aten.mul.Tensor(sub_5, clamp_max_3);  sub_5 = None
        add_8: "f32[1, 128, 320, 478]" = torch.ops.aten.add.Tensor(add_6, mul_11);  add_6 = mul_11 = None
        convert_element_type_7: "bf16[1, 128, 320, 478]" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None
        constant_pad_nd: "bf16[1, 128, 320, 479]" = torch.ops.aten.constant_pad_nd.default(convert_element_type_7, [0, 1, 0, 0], 0.0);  convert_element_type_7 = None
        cat: "bf16[1, 256, 320, 479]" = torch.ops.aten.cat.default([arg5_1, constant_pad_nd], 1);  arg5_1 = constant_pad_nd = None
        copy_: "f32[128]" = torch.ops.aten.copy_.default(arg1_1, add_1);  arg1_1 = add_1 = None
        copy__1: "f32[128]" = torch.ops.aten.copy_.default(arg2_1, add_2);  arg2_1 = add_2 = None
        return (getitem_1, rsqrt, convert_element_type_4, clamp_max, convert_element_type_6, clamp_max_1, clamp_max_2, clamp_max_3, cat, copy_, copy__1)



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
