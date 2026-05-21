"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_infer_000
Pattern hash: bdb10d34fe50
Shape hash: fa5e91d1
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
_shapes_config = "(T([12000, 384], f16), T([8, 1500, 384], f16, stride=(576000, 1, 1500)), T([384], f16), T([384], f16), S([8, 1500, 384]), S([12000, 384]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_19: "f16[12000, 384]", add_26: "f16[8, 1500, 384]", arg66_1: "f16[384]", arg67_1: "f16[384]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f16[8, 1500, 384]" = torch.ops.aten.view.default(addmm_19, _shape_param_0);  addmm_19 = _shape_param_0 = None
        add_tensor: "f16[8, 1500, 384]" = torch.ops.aten.add.Tensor(add_26, view_default);  add_26 = view_default = None
        convert_element_type_default: "f32[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        clamp_min_default: "f32[8, 1500, 384]" = torch.ops.aten.clamp_min.default(convert_element_type_default, -64504.0);  convert_element_type_default = None
        clamp_max_default: "f32[8, 1500, 384]" = torch.ops.aten.clamp_max.default(clamp_min_default, 64504.0);  clamp_min_default = None
        convert_element_type_default_1: "f16[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(clamp_max_default, torch.float16);  clamp_max_default = None
        clone_default: "f16[8, 1500, 384]" = torch.ops.aten.clone.default(convert_element_type_default_1, memory_format = torch.contiguous_format);  convert_element_type_default_1 = None
        convert_element_type_default_2: "f32[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(clone_default, torch.float32);  clone_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1500, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 1500, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, getitem_1);  convert_element_type_default_2 = getitem_1 = None
        add_tensor_1: "f32[8, 1500, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[8, 1500, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_tensor, arg66_1);  mul_tensor = arg66_1 = None
        add_tensor_2: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg67_1);  mul_tensor_1 = arg67_1 = None
        convert_element_type_default_3: "f16[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float16);  add_tensor_2 = None
        view_default_1: "f16[12000, 384]" = torch.ops.aten.view.default(convert_element_type_default_3, _shape_param_1);  convert_element_type_default_3 = _shape_param_1 = None
        return view_default_1



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
