"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_train
Pattern hash: 56c27cd60aa2
Shape hash: af50fef5
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 512], f32), T([64], i64, max=64), T([8, 1024, 512], f32), T([512], f32), T([512, 512], f32), T([512, 512], f32), S([8, 1024, 512]), S([8192, 512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_85: "f32[8192, 512]", inductor_seeds_default: "i64[64]", add_84: "f32[8, 1024, 512]", primals_120: "f32[512]", primals_121: "f32[512, 512]", primals_122: "f32[512, 512]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_85, _shape_param_0);  mm_85 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 56);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_84, mul_tensor_1);  add_84 = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_120, mul_tensor_2);  primals_120 = mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default: "f32[512, 512]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
        reshape_default_1: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_1);  mul_tensor_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_1: "f32[512, 512]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
        return (permute_default, reshape_default_1, permute_default_1)


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
