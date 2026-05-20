"""
Standalone repro captured via capture_hook.
Label: hf_MegatronBertForCausalLM_train
Pattern hash: ad58083c0d01
Shape hash: 6cf7b975
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([29056, 1024], f32), T([16, 512], i64, max=29056), T([2, 1024], f32), T([512, 1024], f32), T([1, 512], i64, max=512), T([1024], f32), T([1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), S([8192, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, primals_3: "f32[29056, 1024]", primals_2: "i64[16, 512]", primals_5: "f32[2, 1024]", primals_6: "f32[512, 1024]", primals_4: "i64[1, 512]", primals_7: "f32[1024]", primals_8: "f32[1024]", primals_9: "f32[1024, 1024]", primals_11: "f32[1024, 1024]", primals_13: "f32[1024, 1024]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:628 in forward, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        full_default: "i64[16, 512]" = torch.ops.aten.full.default([16, 512], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:89 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding_default: "f32[16, 512, 1024]" = torch.ops.aten.embedding.default(primals_3, primals_2, 0);  primals_3 = primals_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:90 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_default_1: "f32[16, 512, 1024]" = torch.ops.aten.embedding.default(primals_5, full_default);  primals_5 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:91 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add_tensor: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:93 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_default_2: "f32[1, 512, 1024]" = torch.ops.aten.embedding.default(primals_6, primals_4);  primals_6 = primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:94 in forward, code: embeddings += position_embeddings
        add_tensor_1: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_2);  add_tensor = embedding_default_2 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[49]" = torch.ops.prims.inductor_seeds.default(49, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:98 in forward, code: embeddings = self.dropout(embeddings)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_scalar, add_tensor_1);  gt_scalar = add_tensor_1 = None
        mul_tensor_1: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(mul_tensor_1, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[16, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_1, getitem_1);  mul_tensor_1 = getitem_1 = None
        mul_tensor_2: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_3: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, primals_7);  mul_tensor_2 = primals_7 = None
        add_tensor_3: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_3, primals_8);  mul_tensor_3 = primals_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        reshape_default: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_0);  add_tensor_3 = _shape_param_0 = None
        permute_default: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_2: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        return (reshape_default, permute_default, permute_default_1, permute_default_2)


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
