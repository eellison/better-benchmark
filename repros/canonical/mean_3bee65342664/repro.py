"""
Standalone repro captured via capture_hook.
Label: torchbench_llama_infer_000
Pattern hash: 3bee65342664
Shape hash: 90b1e79f
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([32000, 512], f32), T([32, 32], i32, gen=Index(32000)), T([512], f32), S([1024, 512]), S([1024, 512]), S([1024, 512]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[32000, 512]", arg0_1: "i32[32, 32]", arg3_1: "f32[512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        embedding_default: "f32[32, 32, 512]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        pow_tensor_scalar: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(embedding_default, 2)
        mean_dim: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-05);  mean_dim = None
        rsqrt_default: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(embedding_default, rsqrt_default);  embedding_default = rsqrt_default = None
        mul_tensor_1: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg3_1);  mul_tensor = arg3_1 = None
        view_default: "f32[1024, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_0);  _shape_param_0 = None
        view_default_1: "f32[1024, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_1);  _shape_param_1 = None
        view_default_2: "f32[1024, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_2);  mul_tensor_1 = _shape_param_2 = None
        return (view_default, view_default_1, view_default_2)

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
