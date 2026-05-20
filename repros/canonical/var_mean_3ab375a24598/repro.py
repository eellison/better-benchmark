"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_infer
Pattern hash: 3ab375a24598
Shape hash: 22caa6f4
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([30522, 768], f32), T([32, 512], i64, max=30522), T([512, 768], f32), T([1, 512], i64, max=512), T([1, 512], i64, max=2), T([2, 768], f32), T([768], f32), T([768], f32), T([384, 768], f32), S([32, 512]), S([16384, 768]))"

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "f32[30522, 768]", arg0_1: "i64[32, 512]", arg4_1: "f32[512, 768]", arg3_1: "i64[1, 512]", arg1_1: "i64[1, 512]", arg5_1: "f32[2, 768]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg15_1: "f32[384, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:99 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding_default: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg2_1, arg0_1, 0);  arg2_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:100 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_default_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, arg3_1);  arg4_1 = arg3_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:103 in forward, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:628 in forward, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_default: "i64[32, 512]" = torch.ops.aten.expand.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:101 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_default_2: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, expand_default);  arg5_1 = expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:103 in forward, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        add_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_2);  add_tensor = embedding_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:104 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1);  add_tensor_1 = getitem_1 = None
        add_tensor_2: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg6_1);  mul_tensor = arg6_1 = None
        add_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg7_1);  mul_tensor_1 = arg7_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        reshape_default: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_1);  _shape_param_1 = None
        permute_default: "f32[768, 384]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_default_1: "f32[32, 768, 512]" = torch.ops.aten.permute.default(add_tensor_3, [0, 2, 1]);  add_tensor_3 = None
        return (reshape_default, permute_default, permute_default_1)


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
