"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_DistilBert_train
Pattern hash: 386e29efa04a
Shape hash: 939b0e30
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([30522, 768], f32), T([8, 512], i64, gen=Index(30522)), T([512, 768], f32), T([1, 512], i64, gen=Index(512)), T([768], f32), T([768], f32), S([4096, 768]))"

class Repro(torch.nn.Module):
    def forward(self, primals_2: "f32[30522, 768]", primals_1: "i64[8, 512]", primals_4: "f32[512, 768]", primals_3: "i64[1, 512]", primals_5: "f32[768]", primals_6: "f32[768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:103 in forward, code: inputs_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        embedding_default: "f32[8, 512, 768]" = torch.ops.aten.embedding.default(primals_2, primals_1, 0);  primals_2 = primals_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:117 in forward, code: position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
        embedding_default_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_4, primals_3);  primals_4 = primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:119 in forward, code: embeddings = inputs_embeds + position_embeddings  # (bs, max_seq_length, dim)
        add_tensor: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:120 in forward, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_5);  mul_tensor = primals_5 = None
        add_tensor_2: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_6);  mul_tensor_1 = primals_6 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[13]" = torch.ops.prims.inductor_seeds.default(13, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:121 in forward, code: embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 512, 768]" = torch.ops.prims.inductor_random.default([8, 512, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor_2: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, add_tensor_2);  gt_scalar = add_tensor_2 = None
        mul_tensor_3: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[4096, 768]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_0);  mul_tensor_3 = _shape_param_0 = None
        return reshape_default



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
