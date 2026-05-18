"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_training
Pattern hash: aa06d23f7965
Shape hash: 1f6bb215
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_34: "f32[8192, 2048]", _shape_param_0, inductor_seeds_default: "i64[64]", primals_51: "f32[512, 2048]", _shape_param_1, bmm_13: "f32[64, 1024, 64]", _shape_param_2, _shape_param_3, primals_59: "f32[512, 512]", _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        reshape_default: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_34, _shape_param_0);  mm_34 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_default: "f32[8, 1024, 2048]" = torch.ops.aten.relu.default(reshape_default);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_scalar, relu_default);  gt_scalar = relu_default = None
        mul_tensor_1: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        reshape_default_1: "f32[8192, 2048]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_2: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_13, _shape_param_2);  bmm_13 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_1: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        clone_default: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        reshape_default_3: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_2: "f32[512, 512]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
        reshape_default_4: "f32[8192, 512]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None
        return (permute_default, reshape_default_1, permute_default_2, reshape_default_4)


def _default_make_inputs():
    return [
    torch.randn([8192, 2048], dtype=torch.float32, device='cuda'),
    [8, 1024, 2048],  # _shape_param_0
    torch.randint(0, 2, [64], dtype=torch.int64, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    [8192, 2048],  # _shape_param_1
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    [8, 8, 1024, 64],  # _shape_param_2
    [8, 1024, -1],  # _shape_param_3
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    [8192, 512],  # _shape_param_4
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
