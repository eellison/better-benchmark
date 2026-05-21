"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train_000
Pattern hash: 4c9e3e3af443
Shape hash: 0fee26a1
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
_shapes_config = "(T([32768, 512], f32), T([512], f32), T([512], f32), S([256, 128, 512]), S([32768, 512]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_361: "f32[32768, 512]", arg1115_1: "f32[512]", arg1116_1: "f32[512]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[256, 128, 512]" = torch.ops.aten.view.default(addmm_361, _shape_param_0);  addmm_361 = _shape_param_0 = None
        relu_default: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_default);  view_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(relu_default, [2], correction = 0, keepdim = True)
        getitem: "f32[256, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[256, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[256, 128, 512]" = torch.ops.aten.sub.Tensor(relu_default, getitem_1);  relu_default = getitem_1 = None
        mul_tensor: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg1115_1);  mul_tensor = arg1115_1 = None
        add_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg1116_1);  mul_tensor_1 = arg1116_1 = None
        view_default_1: "f32[32768, 512]" = torch.ops.aten.view.default(add_tensor_1, _shape_param_1);  add_tensor_1 = _shape_param_1 = None
        permute_default: "f32[512, 32768]" = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        return permute_default



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
