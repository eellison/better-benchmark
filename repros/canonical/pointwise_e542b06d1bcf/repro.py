"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_inference
Pattern hash: e542b06d1bcf
Shape hash: b51c94a3
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
    def forward(self, addmm: "f32[1024, 512]", _shape_param_0, arg1_1: "i64[1, 512]", arg5_1: "f32[512, 512]", arg6_1: "f32[2, 512]", arg7_1: "f32[512]", arg8_1: "f32[512]", _shape_param_1, arg13_1: "f32[128, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:132 in forward, code: inputs_embeds = self.embedding_transformation(inputs_embeds)
        reshape_default: "f32[8, 128, 512]" = torch.ops.aten.reshape.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:108 in forward, code: position_ids = self.position_ids[:, :seq_length]
        slice_tensor: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg1_1, 1, 0, 128);  arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:136 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_default: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(arg5_1, slice_tensor);  arg5_1 = slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:138 in forward, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        add_tensor: "f32[8, 128, 512]" = torch.ops.aten.add.Tensor(reshape_default, embedding_default);  reshape_default = embedding_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:111 in forward, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        full_default: "i64[8, 128]" = torch.ops.aten.full.default([8, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:137 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_default_1: "f32[8, 128, 512]" = torch.ops.aten.embedding.default(arg6_1, full_default);  arg6_1 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:138 in forward, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        add_tensor_1: "f32[8, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_1);  add_tensor = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_tensor: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_1, arg7_1);  add_tensor_1 = arg7_1 = None
        add_tensor_2: "f32[8, 128, 512]" = torch.ops.aten.add.Tensor(mul_tensor, arg8_1);  mul_tensor = arg8_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        reshape_default_1: "f32[1024, 512]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[512, 128]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        return (reshape_default_1, permute_default)


def _default_make_inputs():
    return [
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 512],  # _shape_param_0
    torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_1
    torch.randn([128, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
