"""
Standalone repro captured via capture_hook.
Label: hf_Reformer_training
Pattern hash: 8f51c0011289
Shape hash: 938d01f1
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm: "f32[32768, 512]", _shape_param_0, _shape_param_1, arg5_1: "f32[256, 512]"):
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1437 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 4096, 512]" = torch.ops.prims.inductor_random.default([8, 4096, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 4096, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.05);  inductor_random_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1436 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[8, 4096, 512]" = torch.ops.aten.reshape.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1437 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_tensor: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.0526315789473684);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1438 in forward, code: hidden_states = self.act_fn(hidden_states)
        relu_default: "f32[8, 4096, 512]" = torch.ops.aten.relu.default(mul_tensor_1);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1450 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_1: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_default, _shape_param_1);  relu_default = _shape_param_1 = None
        permute_default: "f32[512, 256]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        return (reshape_default_1, permute_default)


def _default_make_inputs():
    return [
    torch.randn([32768, 512], dtype=torch.float32, device='cuda'),
    [8, 4096, 512],  # _shape_param_0
    [32768, 512],  # _shape_param_1
    torch.randn([256, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
