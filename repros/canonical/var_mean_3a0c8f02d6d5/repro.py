"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_CycleGAN_and_pix2pix_infer
Pattern hash: 3a0c8f02d6d5
Shape hash: 1b4e0bc8
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
    def forward(self, arg0_1: "bf16[1, 256, 64, 64]"):
        # No stacktrace found for following nodes
        convert_element_type: "f32[1, 256, 64, 64]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type = None
        getitem: "f32[1, 256, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 256, 1, 1]" = var_mean[1];  var_mean = None
        sub: "f32[1, 256, 64, 64]" = torch.ops.aten.sub.Tensor(arg0_1, getitem_1);  arg0_1 = getitem_1 = None
        add: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        convert_element_type_1: "bf16[1, 256, 64, 64]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        relu: "bf16[1, 256, 64, 64]" = torch.ops.aten.relu.default(convert_element_type_1);  convert_element_type_1 = None
        convert_element_type_2: "f32[1, 256, 64, 64]" = torch.ops.prims.convert_element_type.default(relu, torch.float32)
        iota: "i64[66]" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_1: "i64[66]" = torch.ops.aten.abs.default(iota);  iota = None
        sub_1: "i64[66]" = torch.ops.aten.sub.Tensor(63, abs_1);  abs_1 = None
        abs_2: "i64[66]" = torch.ops.aten.abs.default(sub_1);  sub_1 = None
        sub_2: "i64[66]" = torch.ops.aten.sub.Tensor(63, abs_2);  abs_2 = None
        _unsafe_index: "f32[1, 256, 66, 64]" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_2, [None, None, sub_2, None]);  convert_element_type_2 = sub_2 = None
        iota_1: "i64[66]" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_3: "i64[66]" = torch.ops.aten.abs.default(iota_1);  iota_1 = None
        sub_3: "i64[66]" = torch.ops.aten.sub.Tensor(63, abs_3);  abs_3 = None
        abs_4: "i64[66]" = torch.ops.aten.abs.default(sub_3);  sub_3 = None
        sub_4: "i64[66]" = torch.ops.aten.sub.Tensor(63, abs_4);  abs_4 = None
        _unsafe_index_1: "f32[1, 256, 66, 66]" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index, [None, None, None, sub_4]);  _unsafe_index = sub_4 = None
        convert_element_type_3: "bf16[1, 256, 66, 66]" = torch.ops.prims.convert_element_type.default(_unsafe_index_1, torch.bfloat16);  _unsafe_index_1 = None
        return (relu, convert_element_type_3)



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
