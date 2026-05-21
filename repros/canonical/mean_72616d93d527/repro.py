"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_infer_000
Pattern hash: 72616d93d527
Shape hash: 3c8bda83
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
_shapes_config = "(T([4096, 512], f32), T([32, 128, 512], f32), T([512], f32), S([32, 128, 512]), S([4096, 512]), S([4096, 512]), S([4096, 512]), S([4096, 512]), S([4096, 512]), S([4096, 512]), S([4096, 512]), S([4096, 512]), S([4096, 512]), S([4096, 512]), S([4096, 512]), S([4096, 512]), S([4096, 512]), S([4096, 512]), S([4096, 512]), S([4096, 512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_60: "f32[4096, 512]", add_67: "f32[32, 128, 512]", arg75_1: "f32[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16):
        # No stacktrace found for following nodes
        view_default: "f32[32, 128, 512]" = torch.ops.aten.view.default(mm_60, _shape_param_0);  mm_60 = _shape_param_0 = None
        add_tensor: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_67, view_default);  add_67 = view_default = None
        pow_tensor_scalar: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor_1: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg75_1, mul_tensor);  arg75_1 = mul_tensor = None
        view_default_1: "f32[4096, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_1);  _shape_param_1 = None
        view_default_2: "f32[4096, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_2);  _shape_param_2 = None
        view_default_3: "f32[4096, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_3);  _shape_param_3 = None
        view_default_4: "f32[4096, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_4);  _shape_param_4 = None
        view_default_5: "f32[4096, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_5);  _shape_param_5 = None
        view_default_6: "f32[4096, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_6);  _shape_param_6 = None
        view_default_7: "f32[4096, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_7);  _shape_param_7 = None
        view_default_8: "f32[4096, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_8);  _shape_param_8 = None
        view_default_9: "f32[4096, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_9);  _shape_param_9 = None
        view_default_10: "f32[4096, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_10);  _shape_param_10 = None
        view_default_11: "f32[4096, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_11);  _shape_param_11 = None
        view_default_12: "f32[4096, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_12);  _shape_param_12 = None
        view_default_13: "f32[4096, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_13);  _shape_param_13 = None
        view_default_14: "f32[4096, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_14);  _shape_param_14 = None
        view_default_15: "f32[4096, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_15);  _shape_param_15 = None
        view_default_16: "f32[4096, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_16);  mul_tensor_1 = _shape_param_16 = None
        return (view_default_1, view_default_2, view_default_3, view_default_4, view_default_5, view_default_6, view_default_7, view_default_8, view_default_9, view_default_10, view_default_11, view_default_12, view_default_13, view_default_14, view_default_15, view_default_16)



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
