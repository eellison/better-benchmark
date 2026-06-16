"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_unet_infer
Pattern hash: 783d1118b332
Shape hash: a2b68844
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
    def forward(self, arg0_1: "bf16[512]", arg1_1: "bf16[1, 512, 40, 59]", arg2_1: "bf16[512]", arg3_1: "bf16[512]", arg4_1: "bf16[512]", arg5_1: "bf16[1, 512, 80, 119]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "f32[512]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        unsqueeze: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
        unsqueeze_1: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        sub: "f32[1, 512, 40, 59]" = torch.ops.aten.sub.Tensor(arg1_1, unsqueeze_1);  arg1_1 = unsqueeze_1 = None
        convert_element_type_1: "f32[512]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        add: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
        sqrt: "f32[512]" = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_2: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
        unsqueeze_3: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_1: "f32[1, 512, 40, 59]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_4: "bf16[512, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_5: "bf16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_2: "f32[1, 512, 40, 59]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        unsqueeze_6: "bf16[512, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_7: "bf16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1: "f32[1, 512, 40, 59]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
        convert_element_type_2: "bf16[1, 512, 40, 59]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        relu: "bf16[1, 512, 40, 59]" = torch.ops.aten.relu.default(convert_element_type_2);  convert_element_type_2 = None
        convert_element_type_3: "f32[1, 512, 40, 59]" = torch.ops.prims.convert_element_type.default(relu, torch.float32);  relu = None
        iota: "i64[80]" = torch.ops.prims.iota.default(80, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_4: "f32[80]" = torch.ops.prims.convert_element_type.default(iota, torch.float32);  iota = None
        mul_3: "f32[80]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 0.4936708860759494);  convert_element_type_4 = None
        clamp_min: "f32[80]" = torch.ops.aten.clamp_min.default(mul_3, 0.0);  mul_3 = None
        view: "f32[80, 1]" = torch.ops.aten.view.default(clamp_min, _shape_param_0);  clamp_min = _shape_param_0 = None
        convert_element_type_5: "i64[80, 1]" = torch.ops.prims.convert_element_type.default(view, torch.int64)
        add_2: "i64[80, 1]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1)
        clamp_max: "i64[80, 1]" = torch.ops.aten.clamp_max.default(add_2, 39);  add_2 = None
        iota_1: "i64[118]" = torch.ops.prims.iota.default(118, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_6: "f32[118]" = torch.ops.prims.convert_element_type.default(iota_1, torch.float32);  iota_1 = None
        mul_4: "f32[118]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.49572649572649574);  convert_element_type_6 = None
        clamp_min_1: "f32[118]" = torch.ops.aten.clamp_min.default(mul_4, 0.0);  mul_4 = None
        convert_element_type_7: "i64[118]" = torch.ops.prims.convert_element_type.default(clamp_min_1, torch.int64)
        add_3: "i64[118]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1)
        clamp_max_1: "i64[118]" = torch.ops.aten.clamp_max.default(add_3, 58);  add_3 = None
        _unsafe_index: "f32[1, 512, 80, 118]" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_3, [None, None, clamp_max, clamp_max_1])
        _unsafe_index_1: "f32[1, 512, 80, 118]" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_3, [None, None, clamp_max, convert_element_type_7]);  clamp_max = None
        sub_1: "f32[1, 512, 80, 118]" = torch.ops.aten.sub.Tensor(_unsafe_index, _unsafe_index_1);  _unsafe_index = None
        sub_2: "f32[118]" = torch.ops.aten.sub.Tensor(clamp_min_1, convert_element_type_7);  clamp_min_1 = None
        clamp_min_2: "f32[118]" = torch.ops.aten.clamp_min.default(sub_2, 0.0);  sub_2 = None
        clamp_max_2: "f32[118]" = torch.ops.aten.clamp_max.default(clamp_min_2, 1.0);  clamp_min_2 = None
        mul_5: "f32[1, 512, 80, 118]" = torch.ops.aten.mul.Tensor(sub_1, clamp_max_2);  sub_1 = None
        add_4: "f32[1, 512, 80, 118]" = torch.ops.aten.add.Tensor(_unsafe_index_1, mul_5);  _unsafe_index_1 = mul_5 = None
        _unsafe_index_2: "f32[1, 512, 80, 118]" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_3, [None, None, convert_element_type_5, clamp_max_1]);  clamp_max_1 = None
        _unsafe_index_3: "f32[1, 512, 80, 118]" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_3, [None, None, convert_element_type_5, convert_element_type_7]);  convert_element_type_3 = convert_element_type_7 = None
        sub_3: "f32[1, 512, 80, 118]" = torch.ops.aten.sub.Tensor(_unsafe_index_2, _unsafe_index_3);  _unsafe_index_2 = None
        mul_6: "f32[1, 512, 80, 118]" = torch.ops.aten.mul.Tensor(sub_3, clamp_max_2);  sub_3 = clamp_max_2 = None
        add_5: "f32[1, 512, 80, 118]" = torch.ops.aten.add.Tensor(_unsafe_index_3, mul_6);  _unsafe_index_3 = mul_6 = None
        sub_4: "f32[1, 512, 80, 118]" = torch.ops.aten.sub.Tensor(add_4, add_5);  add_4 = None
        sub_5: "f32[80, 1]" = torch.ops.aten.sub.Tensor(view, convert_element_type_5);  view = convert_element_type_5 = None
        clamp_min_3: "f32[80, 1]" = torch.ops.aten.clamp_min.default(sub_5, 0.0);  sub_5 = None
        clamp_max_3: "f32[80, 1]" = torch.ops.aten.clamp_max.default(clamp_min_3, 1.0);  clamp_min_3 = None
        mul_7: "f32[1, 512, 80, 118]" = torch.ops.aten.mul.Tensor(sub_4, clamp_max_3);  sub_4 = clamp_max_3 = None
        add_6: "f32[1, 512, 80, 118]" = torch.ops.aten.add.Tensor(add_5, mul_7);  add_5 = mul_7 = None
        convert_element_type_8: "bf16[1, 512, 80, 118]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        constant_pad_nd: "bf16[1, 512, 80, 119]" = torch.ops.aten.constant_pad_nd.default(convert_element_type_8, [0, 1, 0, 0], 0.0);  convert_element_type_8 = None
        cat: "bf16[1, 1024, 80, 119]" = torch.ops.aten.cat.default([arg5_1, constant_pad_nd], 1);  arg5_1 = constant_pad_nd = None
        return cat



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
