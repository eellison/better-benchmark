"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-0.6B_000
Pattern hash: 2980f3ec2954
Shape hash: fd9e3b66
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
_shapes_config = "(T([151936, 1024], bf16), T([4, 512], i64, gen=Index(151936)), T([1024], bf16), S([2048, 1024]), S([2048, 1024]), S([2048, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "bf16[151936, 1024]", arg0_1: "i64[4, 512]", arg3_1: "bf16[1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        embedding_default: "bf16[4, 512, 1024]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        convert_element_type_default: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(embedding_default, torch.float32);  embedding_default = None
        pow_tensor_scalar: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None
        convert_element_type_default_1: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg3_1, convert_element_type_default_1);  arg3_1 = convert_element_type_default_1 = None
        view_default: "bf16[2048, 1024]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_0);  _shape_param_0 = None
        view_default_1: "bf16[2048, 1024]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_1);  _shape_param_1 = None
        view_default_2: "bf16[2048, 1024]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_2);  mul_tensor_1 = _shape_param_2 = None
        return (view_default, view_default_2, view_default_1)



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
