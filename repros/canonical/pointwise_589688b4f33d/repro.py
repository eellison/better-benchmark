"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train_001
Pattern hash: 589688b4f33d
Shape hash: 6a76fd57
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
_shapes_config = "(T([4096, 16384], f32), T([4096, 16384], f32), S([8, 512, 16384]), S([8, 512, 16384]), S([4096, 16384]))"

class Repro(torch.nn.Module):
    def forward(self, mm_136: "f32[4096, 16384]", arg24_1: "f32[4096, 16384]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[8, 512, 16384]" = torch.ops.aten.view.default(mm_136, _shape_param_0);  mm_136 = _shape_param_0 = None
        view_default_1: "f32[8, 512, 16384]" = torch.ops.aten.view.default(arg24_1, _shape_param_1);  arg24_1 = _shape_param_1 = None
        mul_tensor: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_default_1, 0.5)
        mul_tensor_1: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_default, mul_tensor);  mul_tensor = None
        pow_tensor_scalar: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_default_1, 3.0)
        mul_tensor_2: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_default_1, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_3: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor_1: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_default, 1.0)
        mul_tensor_4: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_default, add_tensor_1);  view_default = add_tensor_1 = None
        mul_tensor_5: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_tensor_5);  mul_tensor_5 = None
        mul_tensor_6: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_1, sub_tensor);  mul_tensor_1 = sub_tensor = None
        mul_tensor_7: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_6, 0.7978845608028654);  mul_tensor_6 = None
        mul_tensor_8: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.044715)
        pow_tensor_scalar_1: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_default_1, 2.0);  view_default_1 = None
        mul_scalar: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 3.0);  pow_tensor_scalar_1 = None
        mul_tensor_9: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_8, mul_scalar);  mul_tensor_8 = mul_scalar = None
        add_tensor_2: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(mul_tensor_7, mul_tensor_9);  mul_tensor_7 = mul_tensor_9 = None
        mul_tensor_10: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 0.5);  mul_tensor_4 = None
        add_tensor_3: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_10);  add_tensor_2 = mul_tensor_10 = None
        view_default_2: "f32[4096, 16384]" = torch.ops.aten.view.default(add_tensor_3, _shape_param_2);  add_tensor_3 = _shape_param_2 = None
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
