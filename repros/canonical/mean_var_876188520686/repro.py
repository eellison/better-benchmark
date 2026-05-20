"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf-1-6-linux.aws.a100_graph22
Pattern hash: 876188520686
Shape hash: ccb67c2d
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([512, 768], bf16), T([4, 128, 768], bf16), T([768], bf16), T([768], bf16), T([768, 768], bf16), T([768, 768], bf16), S([4, 128, 768]), S([512, 768]), S([512, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_65: "bf16[512, 768]", add_74: "bf16[4, 128, 768]", arg181_1: "bf16[768]", arg182_1: "bf16[768]", arg183_1: "bf16[768, 768]", arg185_1: "bf16[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "bf16[4, 128, 768]" = torch.ops.aten.view.default(addmm_65, _shape_param_0);  addmm_65 = _shape_param_0 = None
        add_tensor: "bf16[4, 128, 768]" = torch.ops.aten.add.Tensor(add_74, view_default);  add_74 = view_default = None
        mean_dim: "bf16[4, 128, 1]" = torch.ops.aten.mean.dim(add_tensor, [-1], True)
        convert_element_type_default: "f32[4, 128, 768]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32)
        var_correction: "f32[4, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_default, [-1], correction = 1.0, keepdim = True);  convert_element_type_default = None
        sqrt_default: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_correction);  var_correction = None
        convert_element_type_default_1: "bf16[4, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_default, torch.bfloat16);  sqrt_default = None
        sub_tensor: "bf16[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_tensor, mean_dim);  add_tensor = mean_dim = None
        mul_tensor: "bf16[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg181_1, sub_tensor);  arg181_1 = sub_tensor = None
        add_tensor_1: "bf16[4, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-06);  convert_element_type_default_1 = None
        div_tensor: "bf16[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_tensor, add_tensor_1);  mul_tensor = add_tensor_1 = None
        add_tensor_2: "bf16[4, 128, 768]" = torch.ops.aten.add.Tensor(div_tensor, arg182_1);  div_tensor = arg182_1 = None
        view_default_1: "bf16[512, 768]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_1);  _shape_param_1 = None
        permute_default: "bf16[768, 768]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        view_default_2: "bf16[512, 768]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_2);  add_tensor_2 = _shape_param_2 = None
        permute_default_1: "bf16[768, 768]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        return (view_default_1, permute_default, view_default_2, permute_default_1)


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
