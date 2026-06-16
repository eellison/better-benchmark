"""
Standalone repro captured via capture_hook.
Label: torchbench_mnasnet1_0_train
Pattern hash: c2c0ad00da48
Shape hash: 441beb73
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
    def forward(self, arg0_1: "bf16[32, 1280, 7, 7]", arg1_1: "f32[1280]", arg2_1: "f32[1280]", arg3_1: "f32[1280]", arg4_1: "f32[1280]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "f32[32, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type = None
        getitem: "f32[1, 1280, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 1280, 1, 1]" = var_mean[1];  var_mean = None
        add: "f32[1, 1280, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 1280, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[32, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(arg0_1, getitem_1);  arg0_1 = None
        mul: "f32[32, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul_1: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze, 0.00029999999999996696);  squeeze = None
        mul_2: "f32[1280]" = torch.ops.aten.mul.Tensor(arg1_1, 0.9997)
        add_1: "f32[1280]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_1: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_1, 1.0006381620931717);  squeeze_1 = None
        mul_4: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_3, 0.00029999999999996696);  mul_3 = None
        mul_5: "f32[1280]" = torch.ops.aten.mul.Tensor(arg2_1, 0.9997)
        add_2: "f32[1280]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_1: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[32, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_3: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_3: "f32[32, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_1: "bf16[32, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        relu: "bf16[32, 1280, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_1);  convert_element_type_1 = None
        mean: "bf16[32, 1280]" = torch.ops.aten.mean.dim(relu, [2, 3]);  relu = None
        inductor_seeds: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[32, 1280]" = torch.ops.prims.inductor_random.default(_shape_param_0, inductor_lookup_seed, 'rand');  _shape_param_0 = inductor_lookup_seed = None
        lt: "b8[32, 1280]" = torch.ops.aten.lt.Scalar(inductor_random, 0.8);  inductor_random = None
        convert_element_type_2: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(lt, torch.bfloat16)
        div: "bf16[32, 1280]" = torch.ops.aten.div.Scalar(convert_element_type_2, 0.8);  convert_element_type_2 = None
        mul_7: "bf16[32, 1280]" = torch.ops.aten.mul.Tensor(mean, div);  mean = div = None
        copy_: "f32[1280]" = torch.ops.aten.copy_.default(arg1_1, add_1);  arg1_1 = add_1 = None
        copy__1: "f32[1280]" = torch.ops.aten.copy_.default(arg2_1, add_2);  arg2_1 = add_2 = None
        return (getitem_1, rsqrt, lt, mul_7, copy_, copy__1)



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
