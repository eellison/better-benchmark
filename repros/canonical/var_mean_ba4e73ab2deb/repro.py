"""
Standalone repro captured via capture_hook.
Label: torchbench_densenet121_train
Pattern hash: ba4e73ab2deb
Shape hash: 30ad209f
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
    def forward(self, arg0_1: "bf16[4, 512, 14, 14]", arg1_1: "f32[512]", arg2_1: "f32[512]", arg3_1: "f32[512]", arg4_1: "f32[512]"):
        # No stacktrace found for following nodes
        avg_pool2d: "bf16[4, 512, 7, 7]" = torch.ops.aten.avg_pool2d.default(arg0_1, [2, 2], [2, 2]);  arg0_1 = None
        convert_element_type: "f32[4, 512, 7, 7]" = torch.ops.prims.convert_element_type.default(avg_pool2d, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type = None
        getitem: "f32[1, 512, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 512, 1, 1]" = var_mean[1];  var_mean = None
        add: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(avg_pool2d, getitem_1)
        mul: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_1: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_1: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
        mul_2: "f32[512]" = torch.ops.aten.mul.Tensor(arg1_1, 0.9)
        add_1: "f32[512]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[512]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.005128205128205);  squeeze_2 = None
        mul_4: "f32[512]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[512]" = torch.ops.aten.mul.Tensor(arg2_1, 0.9)
        add_2: "f32[512]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_1: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_3: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_3: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_1: "bf16[4, 512, 7, 7]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        relu: "bf16[4, 512, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_1);  convert_element_type_1 = None
        unsqueeze_4: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_5: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        unsqueeze_6: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None
        copy_: "f32[512]" = torch.ops.aten.copy_.default(arg1_1, add_1);  arg1_1 = add_1 = None
        copy__1: "f32[512]" = torch.ops.aten.copy_.default(arg2_1, add_2);  arg2_1 = add_2 = None
        return (avg_pool2d, squeeze_1, relu, unsqueeze_6, copy_, copy__1)



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
