"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train
Pattern hash: 99cc7655f03b
Shape hash: 504a3306
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([1, 512], i64, gen=Index(512)), T([32768, 512], f32), T([512, 512], f32), T([2, 512], f32), T([512], f32), T([512], f32), T([128, 512], f32), S([256, 128, 512]), S([32768, 512]))"

class Repro(torch.nn.Module):
    def forward(self, primals_2: "i64[1, 512]", addmm: "f32[32768, 512]", primals_6: "f32[512, 512]", primals_7: "f32[2, 512]", primals_8: "f32[512]", primals_9: "f32[512]", primals_14: "f32[128, 512]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:108 in forward, code: position_ids = self.position_ids[:, :seq_length]
        slice_tensor: "i64[1, 128]" = torch.ops.aten.slice.Tensor(primals_2, 1, 0, 128);  primals_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:111 in forward, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        full_default: "i64[256, 128]" = torch.ops.aten.full.default([256, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:132 in forward, code: inputs_embeds = self.embedding_transformation(inputs_embeds)
        reshape_default: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:136 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_default: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(primals_6, slice_tensor);  primals_6 = slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:137 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_default_1: "f32[256, 128, 512]" = torch.ops.aten.embedding.default(primals_7, full_default);  primals_7 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:138 in forward, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        add_tensor: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(reshape_default, embedding_default);  reshape_default = embedding_default = None
        add_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_1);  add_tensor = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_tensor: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_8);  add_tensor_1 = primals_8 = None
        add_tensor_2: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_tensor, primals_9);  mul_tensor = primals_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        reshape_default_1: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[512, 128]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
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
