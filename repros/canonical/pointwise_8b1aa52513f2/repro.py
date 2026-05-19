"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf_cuda_h100-2-5-linux.aws.h100_graph36
Pattern hash: 8b1aa52513f2
Shape hash: 0095c6c1
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
    def forward(self, addmm_22: "bf16[512, 3072]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "bf16[1, 512, 3072]" = torch.ops.aten.view.default(addmm_22, _shape_param_0);  addmm_22 = _shape_param_0 = None
        mul_tensor: "bf16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_default, 0.5)
        pow_tensor_scalar: "bf16[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_default, 3.0)
        mul_tensor_1: "bf16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "bf16[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_default, mul_tensor_1);  view_default = mul_tensor_1 = None
        mul_tensor_2: "bf16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "bf16[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_tensor_2);  mul_tensor_2 = None
        add_tensor_1: "bf16[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_default, 1.0);  tanh_default = None
        mul_tensor_3: "bf16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor_1);  mul_tensor = add_tensor_1 = None
        view_default_1: "bf16[512, 3072]" = torch.ops.aten.view.default(mul_tensor_3, _shape_param_1);  mul_tensor_3 = _shape_param_1 = None
        return view_default_1


def _default_make_inputs():
    return [
    torch.randn([512, 3072], dtype=torch.bfloat16, device='cuda'),
    [1, 512, 3072],  # _shape_param_0
    [-1, 3072],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
