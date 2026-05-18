"""
Standalone repro captured via capture_hook.
Label: hf_RobertaForCausalLM_inference
Pattern hash: 61b6fdf304a1
Shape hash: feb303b6
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
    def forward(self, arg2_1: "f32[50265, 768]", arg0_1: "i64[8, 512]", arg1_1: "i64[1, 512]", _shape_param_0, cumsum: "i64[8, 512]", convert_element_type: "i32[8, 512]", _shape_param_1, arg3_1: "f32[2, 768]", arg4_1: "f32[512, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:116 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding_default: "f32[8, 512, 768]" = torch.ops.aten.embedding.default(arg2_1, arg0_1, 0);  arg2_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:109 in forward, code: buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
        expand_default: "i64[8, 512]" = torch.ops.aten.expand.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:158 in create_position_ids_from_input_ids, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        convert_element_type_default: "i32[8, 512]" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
        add_tensor: "i32[8, 512]" = torch.ops.aten.add.Tensor(convert_element_type_default, 0);  convert_element_type_default = None
        mul_tensor: "i32[8, 512]" = torch.ops.aten.mul.Tensor(add_tensor, convert_element_type);  add_tensor = convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:159 in create_position_ids_from_input_ids, code: return incremental_indices.long() + padding_idx
        convert_element_type_default_1: "i64[8, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.int64);  mul_tensor = None
        add_tensor_1: "i64[8, 512]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 0);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:110 in forward, code: buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
        gather_default: "i64[8, 512]" = torch.ops.aten.gather.default(expand_default, 1, add_tensor_1);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:111 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_default_1: "i64[8, 512]" = torch.ops.aten.expand.default(gather_default, _shape_param_1);  gather_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:117 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_default_1: "f32[8, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, expand_default_1);  arg3_1 = expand_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:118 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add_tensor_2: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:120 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_default_2: "f32[8, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, add_tensor_1, 0);  arg4_1 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:121 in forward, code: embeddings = embeddings + position_embeddings
        add_tensor_3: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_2, embedding_default_2);  add_tensor_2 = embedding_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:123 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_3, [2], correction = 0, keepdim = True);  add_tensor_3 = None
        getitem: "f32[8, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([50265, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
    [8, -1],  # _shape_param_0
    torch.randint(0, 2, [8, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [8, 512], dtype=torch.int32, device='cuda'),
    [8, 512],  # _shape_param_1
    torch.randn([2, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
