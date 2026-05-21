"""
Standalone repro captured via capture_hook.
Label: torchbench_nvidia_deeprecommender_train
Pattern hash: 921c332e9460
Shape hash: f09cd101
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
    def forward(self, addmm_2: "f32[1024, 1024]", primals_8: "f32[512, 1024]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        gt_scalar: "b8[1024, 1024]" = torch.ops.aten.gt.Scalar(addmm_2, 0)
        mul_tensor: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(addmm_2, 1.0507009873554805)
        mul_tensor_1: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(addmm_2, 1.0);  addmm_2 = None
        expm1_default: "f32[1024, 1024]" = torch.ops.aten.expm1.default(mul_tensor_1);  mul_tensor_1 = None
        mul_tensor_2: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(expm1_default, 1.7580993408473766);  expm1_default = None
        where_self: "f32[1024, 1024]" = torch.ops.aten.where.self(gt_scalar, mul_tensor, mul_tensor_2);  gt_scalar = mul_tensor = mul_tensor_2 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:136 in encode, code: x = self.drop(x)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[1024, 1024]" = torch.ops.prims.inductor_random.default([1024, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar_1: "b8[1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.8);  inductor_random_default = None
        mul_tensor_3: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(gt_scalar_1, where_self);  gt_scalar_1 = where_self = None
        mul_tensor_4: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 5.000000000000001);  mul_tensor_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:160 in decode, code: input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
        permute_default: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        return (mul_tensor_4, permute_default)


def _default_make_inputs():
    return [
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
