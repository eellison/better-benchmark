"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train_001
Pattern hash: e697d67925cd
Shape hash: 524c6a2f
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
_shapes_config = "(T([131072, 288], f32), T([131072, 288], f32), S([512, 256, 288]), S([512, 256, 288]), S([131072, 288]))"

class Repro(torch.nn.Module):
    def forward(self, mm_66: "f32[131072, 288]", arg225_1: "f32[131072, 288]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[512, 256, 288]" = torch.ops.aten.view.default(mm_66, _shape_param_0);  mm_66 = _shape_param_0 = None
        view_default_1: "f32[512, 256, 288]" = torch.ops.aten.view.default(arg225_1, _shape_param_1);  arg225_1 = _shape_param_1 = None
        neg_default: "f32[512, 256, 288]" = torch.ops.aten.neg.default(view_default_1)
        exp_default: "f32[512, 256, 288]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[512, 256, 288]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[512, 256, 288]" = torch.ops.aten.reciprocal.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[512, 256, 288]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_1: "f32[512, 256, 288]" = torch.ops.aten.mul.Tensor(view_default, mul_tensor);  view_default = None
        sub_tensor: "f32[512, 256, 288]" = torch.ops.aten.sub.Tensor(1, mul_tensor);  mul_tensor = None
        mul_tensor_2: "f32[512, 256, 288]" = torch.ops.aten.mul.Tensor(view_default_1, sub_tensor);  view_default_1 = sub_tensor = None
        add_tensor_1: "f32[512, 256, 288]" = torch.ops.aten.add.Tensor(mul_tensor_2, 1);  mul_tensor_2 = None
        mul_tensor_3: "f32[512, 256, 288]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_1);  mul_tensor_1 = add_tensor_1 = None
        view_default_2: "f32[131072, 288]" = torch.ops.aten.view.default(mul_tensor_3, _shape_param_2);  mul_tensor_3 = _shape_param_2 = None
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
