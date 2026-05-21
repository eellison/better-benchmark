"""
Standalone repro captured via capture_hook.
Label: torchbench_llava_infer_000
Pattern hash: ee17938d7aff
Shape hash: af568cdb
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
_shapes_config = "(T([512, 4096], f16), T([1, 512, 4096], f16), T([4096], f16), S([1, 512, 4096]), S([512, 4096]), S([512, 4096]), S([512, 4096]))"

class Repro(torch.nn.Module):
    def forward(self, mm_216: "f16[512, 4096]", add_216: "f16[1, 512, 4096]", arg282_1: "f16[4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f16[1, 512, 4096]" = torch.ops.aten.view.default(mm_216, _shape_param_0);  mm_216 = _shape_param_0 = None
        add_tensor: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_216, view_default);  add_216 = view_default = None
        convert_element_type_default: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        pow_tensor_scalar: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor_1: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None
        convert_element_type_default_1: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float16);  mul_tensor = None
        mul_tensor_1: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg282_1, convert_element_type_default_1);  arg282_1 = convert_element_type_default_1 = None
        view_default_1: "f16[512, 4096]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_1);  _shape_param_1 = None
        view_default_2: "f16[512, 4096]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_2);  _shape_param_2 = None
        view_default_3: "f16[512, 4096]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_3);  mul_tensor_1 = _shape_param_3 = None
        return (view_default_1, view_default_2, view_default_3)



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
