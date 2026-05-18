"""
Standalone repro captured via capture_hook.
Label: hf_t5_base_train
Pattern hash: b088bded9ec8
Shape hash: 47a5e051
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_70: "f32[2048, 3072]", _shape_param_0, inductor_seeds_default: "i64[50]", primals_99: "f32[768, 3072]", _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        reshape_default: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_70, _shape_param_0);  mm_70 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_default: "f32[4, 512, 3072]" = torch.ops.aten.relu.default(reshape_default);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 47);  inductor_seeds_default = None
        inductor_random_default: "f32[4, 512, 3072]" = torch.ops.prims.inductor_random.default([4, 512, 3072], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[4, 512, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(gt_scalar, relu_default);  gt_scalar = relu_default = None
        mul_tensor_1: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
        reshape_default_1: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
        return (permute_default, reshape_default_1)


def _default_make_inputs():
    return [
    torch.randn([2048, 3072], dtype=torch.float32, device='cuda'),
    [4, 512, 3072],  # _shape_param_0
    torch.randint(0, 2, [50], dtype=torch.int64, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    [2048, 3072],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
