"""
Standalone repro captured via capture_hook.
Label: torchbench_nvidia_deeprecommender_infer
Pattern hash: 42f96e751539
Shape hash: dd039aaf
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
    def forward(self, addmm_4: "f32[1024, 512]", arg11_1: "f32[197951, 512]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        gt_scalar: "b8[1024, 512]" = torch.ops.aten.gt.Scalar(addmm_4, 0)
        mul_tensor: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(addmm_4, 1.0507009873554805)
        mul_tensor_1: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(addmm_4, 1.0);  addmm_4 = None
        expm1_default: "f32[1024, 512]" = torch.ops.aten.expm1.default(mul_tensor_1);  mul_tensor_1 = None
        mul_tensor_2: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(expm1_default, 1.7580993408473766);  expm1_default = None
        where_self: "f32[1024, 512]" = torch.ops.aten.where.self(gt_scalar, mul_tensor, mul_tensor_2);  gt_scalar = mul_tensor = mul_tensor_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:160 in decode, code: input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
        permute_default: "f32[512, 197951]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        return (where_self, permute_default)


def _default_make_inputs():
    return [
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([197951, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
