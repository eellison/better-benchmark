"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_training
Pattern hash: 86560c2d181e
Shape hash: 436c4ff6
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, primals_12: "f32[4096, 16384]", div_1: "f32[1, 16, 128, 128]", _shape_param_0, _shape_param_1, bmm_165: "f32[16, 128, 128]", _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_default_1: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_default: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_1, _shape_param_0);  _shape_param_0 = None
        reshape_default: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_default, _shape_param_1);  expand_default = _shape_param_1 = None
        permute_default_2: "f32[16, 128, 128]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None
        reshape_default_1: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_165, _shape_param_2);  bmm_165 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_tensor: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(reshape_default_1, div_1);  reshape_default_1 = None
        sum_dim_int_list: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_default: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_tensor: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(fma_default, 16.0);  fma_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        reshape_default_2: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(div_tensor, _shape_param_3);  div_tensor = _shape_param_3 = None
        return (permute_default_1, permute_default_2, reshape_default_2)


def _default_make_inputs():
    return [
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 128, 128], dtype=torch.float32, device='cuda'),
    [1, 16, 128, 128],  # _shape_param_0
    [16, 128, 128],  # _shape_param_1
    torch.randn([16, 128, 128], dtype=torch.float32, device='cuda'),
    [1, 16, 128, 128],  # _shape_param_2
    [16, 128, 128],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
