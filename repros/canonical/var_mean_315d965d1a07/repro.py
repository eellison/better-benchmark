"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf_cuda_h100-6-7-linux.aws.h100_graph58
Pattern hash: 315d965d1a07
Shape hash: 4768d2c6
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([512, 240], bf16), T([32, 16, 240], bf16), T([240], bf16), T([240], bf16), S([32, 16, 240]), S([8, 4, 16, -1]), S([7680, 4, 2, 2]), S([8, 240, 8, 8]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_35: "bf16[512, 240]", add_144: "bf16[32, 16, 240]", arg261_1: "bf16[240]", arg262_1: "bf16[240]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "bf16[32, 16, 240]" = torch.ops.aten.view.default(addmm_35, _shape_param_0);  addmm_35 = _shape_param_0 = None
        add_tensor: "bf16[32, 16, 240]" = torch.ops.aten.add.Tensor(add_144, view_default);  add_144 = view_default = None
        convert_element_type_default: "f32[32, 16, 240]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 16, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 16, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        mul_tensor: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_tensor, arg261_1);  mul_tensor = arg261_1 = None
        add_tensor_2: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg262_1);  mul_tensor_1 = arg262_1 = None
        convert_element_type_default_1: "bf16[32, 16, 240]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        view_default_1: "bf16[8, 4, 16, 240]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_1);  convert_element_type_default_1 = _shape_param_1 = None
        permute_default: "bf16[8, 240, 16, 4]" = torch.ops.aten.permute.default(view_default_1, [0, 3, 2, 1]);  view_default_1 = None
        clone_default: "bf16[8, 240, 16, 4]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_2: "bf16[7680, 4, 2, 2]" = torch.ops.aten.view.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        permute_default_1: "bf16[7680, 2, 4, 2]" = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default_1: "bf16[7680, 2, 4, 2]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_3: "bf16[8, 240, 8, 8]" = torch.ops.aten.view.default(clone_default_1, _shape_param_3);  clone_default_1 = _shape_param_3 = None
        return view_default_3


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
