"""
Standalone repro captured via capture_hook.
Label: hf_BertForMaskedLM_infer
Pattern hash: 7dfad1b1c261
Shape hash: ae69658f
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([30522, 768], f32), T([32, 512], i64, max=30522), T([1, 512], i64, max=2), T([1, 512], i64, max=512), T([2, 768], f32), T([512, 768], f32), T([768], f32), T([768], f32), T([768, 768], f32), T([768, 768], f32), S([32, 512]), S([16384, 768]), S([16384, 768]))"

class Repro(torch.nn.Module):
    def forward(self, arg3_1: "f32[30522, 768]", arg0_1: "i64[32, 512]", arg2_1: "i64[1, 512]", arg1_1: "i64[1, 512]", arg4_1: "f32[2, 768]", arg5_1: "f32[512, 768]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg8_1: "f32[768, 768]", arg10_1: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:103 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding_default: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, arg0_1, 0);  arg3_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:96 in forward, code: buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
        expand_default: "i64[1, 512]" = torch.ops.aten.expand.default(arg2_1, [1, -1]);  arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:97 in forward, code: buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
        gather_default: "i64[1, 512]" = torch.ops.aten.gather.default(expand_default, 1, arg1_1);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:98 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_default_1: "i64[32, 512]" = torch.ops.aten.expand.default(gather_default, _shape_param_0);  gather_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:104 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_default_1: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, expand_default_1);  arg4_1 = expand_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:105 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:107 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_default_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, arg1_1);  arg5_1 = arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:108 in forward, code: embeddings = embeddings + position_embeddings
        add_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_2);  add_tensor = embedding_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:110 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1);  add_tensor_1 = getitem_1 = None
        add_tensor_2: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg6_1);  mul_tensor = arg6_1 = None
        add_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg7_1);  mul_tensor_1 = arg7_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_1);  _shape_param_1 = None
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_2);  add_tensor_3 = _shape_param_2 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1)


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
