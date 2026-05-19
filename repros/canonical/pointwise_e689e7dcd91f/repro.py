"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-2-5-linux.aws.a100_graph5
Pattern hash: e689e7dcd91f
Shape hash: 54646aef
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_48: "f32[512, 3072]", arg62_1: "f32[512, 3072]", arg7_1: "f32[3072, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_48, _shape_param_0);  mm_48 = _shape_param_0 = None
        view_default_1: "f32[1, 512, 3072]" = torch.ops.aten.view.default(arg62_1, _shape_param_1);  arg62_1 = _shape_param_1 = None
        mul_tensor: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_default_1, 0.5)
        mul_tensor_1: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_default, mul_tensor);  mul_tensor = None
        pow_tensor_scalar: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_default_1, 3.0)
        mul_tensor_2: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_default_1, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_3: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor_1: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_default, 1.0)
        mul_tensor_4: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_default, add_tensor_1);  view_default = add_tensor_1 = None
        mul_tensor_5: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_tensor_5);  mul_tensor_5 = None
        mul_tensor_6: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_1, sub_tensor);  mul_tensor_1 = sub_tensor = None
        mul_tensor_7: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_6, 0.7978845608028654);  mul_tensor_6 = None
        mul_tensor_8: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.044715)
        pow_tensor_scalar_1: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_default_1, 2.0);  view_default_1 = None
        mul_scalar: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 3.0);  pow_tensor_scalar_1 = None
        mul_tensor_9: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_8, mul_scalar);  mul_tensor_8 = mul_scalar = None
        add_tensor_2: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_tensor_7, mul_tensor_9);  mul_tensor_7 = mul_tensor_9 = None
        mul_tensor_10: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 0.5);  mul_tensor_4 = None
        add_tensor_3: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_10);  add_tensor_2 = mul_tensor_10 = None
        view_default_2: "f32[512, 3072]" = torch.ops.aten.view.default(add_tensor_3, _shape_param_2);  add_tensor_3 = _shape_param_2 = None
        permute_default: "f32[768, 3072]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        permute_default_1: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (view_default_2, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    [1, 512, 3072],  # _shape_param_0
    [1, 512, 3072],  # _shape_param_1
    [512, 3072],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
