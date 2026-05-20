"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-5-5-linux.aws.a100_graph11
Pattern hash: bbc2bd53d27c
Shape hash: 9c8d47f2
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([512, 1024], bf16), T([512, 1, 1024], bf16), T([1024], bf16), T([1024], bf16), S([512, 1, 1, 1, 1024]), S([512, 1, 1024]), S([512, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, mm_103: "bf16[512, 1024]", convert_element_type_186: "bf16[512, 1, 1024]", _param_constant187: "bf16[1024]", _param_constant188: "bf16[1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        unsqueeze_default: "bf16[1, 512, 1024]" = torch.ops.aten.unsqueeze.default(mm_103, 0);  mm_103 = None
        view_default: "bf16[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        permute_default: "bf16[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_default, [0, 3, 4, 1, 2]);  view_default = None
        view_default_1: "bf16[512, 1, 1024]" = torch.ops.aten.view.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None
        add_tensor: "bf16[512, 1, 1024]" = torch.ops.aten.add.Tensor(view_default_1, convert_element_type_186);  view_default_1 = convert_element_type_186 = None
        convert_element_type_default: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[512, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        mul_tensor: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, _param_constant187);  mul_tensor = _param_constant187 = None
        add_tensor_2: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, _param_constant188);  mul_tensor_1 = _param_constant188 = None
        convert_element_type_default_1: "bf16[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        view_default_2: "bf16[512, 1024]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_2);  convert_element_type_default_1 = _shape_param_2 = None
        return view_default_2


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
