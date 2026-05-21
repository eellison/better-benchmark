"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: b04f31a62815
Shape hash: 89563b01
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
_shapes_config = "(T([16384, 3072], f32), T([61], i64), T([768, 3072], f32), S([128, 128, 3072]), S([16384, 3072]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_70: "f32[16384, 3072]", inductor_seeds_default: "i64[61]", primals_196: "f32[768, 3072]", _shape_param_0, _shape_param_1):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        reshape_default: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_70, _shape_param_0);  addmm_70 = _shape_param_0 = None
        mul_tensor: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, 0.5)
        mul_tensor_1: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, 0.7071067811865476);  reshape_default = None
        erf_default: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 58);  inductor_seeds_default = None
        inductor_random_default: "f32[128, 128, 3072]" = torch.ops.prims.inductor_random.default([128, 128, 3072], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[128, 128, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor_3: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(gt_scalar, mul_tensor_2);  gt_scalar = mul_tensor_2 = None
        mul_tensor_4: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 1.1111111111111112);  mul_tensor_3 = None
        reshape_default_1: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_tensor_4, _shape_param_1);  mul_tensor_4 = _shape_param_1 = None
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
        return (reshape_default_1, permute_default)



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
