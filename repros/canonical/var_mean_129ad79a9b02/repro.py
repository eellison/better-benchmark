"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_CycleGAN_and_pix2pix_infer
Pattern hash: 129ad79a9b02
Shape hash: a9dd97a3
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
    def forward(self, arg0_1: "bf16[1, 128, 128, 128]"):
        # No stacktrace found for following nodes
        convert_element_type: "f32[1, 128, 128, 128]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type = None
        getitem: "f32[1, 128, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 128, 1, 1]" = var_mean[1];  var_mean = None
        sub: "f32[1, 128, 128, 128]" = torch.ops.aten.sub.Tensor(arg0_1, getitem_1);  arg0_1 = getitem_1 = None
        add: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        convert_element_type_1: "bf16[1, 128, 128, 128]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        relu: "bf16[1, 128, 128, 128]" = torch.ops.aten.relu.default(convert_element_type_1);  convert_element_type_1 = None
        return relu



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
