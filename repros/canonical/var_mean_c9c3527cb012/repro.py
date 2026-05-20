"""
Standalone repro captured via capture_hook.
Label: hf_DistilBertForMaskedLM_infer
Pattern hash: c9c3527cb012
Shape hash: 70509aa4
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([30522, 768], f32), T([256, 128], i64, gen=Index(30522)), T([1, 512], i64, gen=Index(512)), T([512, 768], f32), T([768], f32), T([768], f32), T([768, 768], f32), T([768, 768], f32), S([32768, 768]), S([32768, 768]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[30522, 768]", arg0_1: "i64[256, 128]", arg2_1: "i64[1, 512]", arg3_1: "f32[512, 768]", arg4_1: "f32[768]", arg5_1: "f32[768]", arg6_1: "f32[768, 768]", arg8_1: "f32[768, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:103 in forward, code: inputs_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        embedding_default: "f32[256, 128, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:112 in forward, code: position_ids = self.position_ids[:, :seq_length]
        slice_tensor: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, 128);  arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:117 in forward, code: position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
        embedding_default_1: "f32[1, 128, 768]" = torch.ops.aten.embedding.default(arg3_1, slice_tensor);  arg3_1 = slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:119 in forward, code: embeddings = inputs_embeds + position_embeddings  # (bs, max_seq_length, dim)
        add_tensor: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:120 in forward, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[256, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[256, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg4_1);  mul_tensor = arg4_1 = None
        add_tensor_2: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg5_1);  mul_tensor_1 = arg5_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[32768, 768]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_0);  _shape_param_0 = None
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[32768, 768]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
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
