"""
Standalone repro captured via capture_hook.
Label: hf_LayoutLMForMaskedLM_train
Pattern hash: bb08ae31fc54
Shape hash: 56b134ce
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([30522, 768], f32), T([32, 512], i64, max=30522), T([512, 768], f32), T([1, 512], i64, max=512), T([1024, 768], f32), T([1024, 768], f32), T([1024, 768], f32), T([1024, 768], f32), T([2, 768], f32), T([768], f32), T([768], f32), T([768, 768], f32), T([768, 768], f32), T([768, 768], f32), S([16384, 768]))"

class Repro(torch.nn.Module):
    def forward(self, primals_2: "f32[30522, 768]", primals_1: "i64[32, 512]", primals_4: "f32[512, 768]", primals_3: "i64[1, 512]", primals_5: "f32[1024, 768]", primals_6: "f32[1024, 768]", primals_7: "f32[1024, 768]", primals_8: "f32[1024, 768]", primals_9: "f32[2, 768]", primals_10: "f32[768]", primals_11: "f32[768]", primals_12: "f32[768, 768]", primals_14: "f32[768, 768]", primals_16: "f32[768, 768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:498 in forward, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        full_default: "i64[32, 512]" = torch.ops.aten.full.default([32, 512], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:501 in forward, code: bbox = torch.zeros(input_shape + (4,), dtype=torch.long, device=device)
        full_default_1: "i64[32, 512, 4]" = torch.ops.aten.full.default([32, 512, 4], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:92 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding_default: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_2, primals_1, 0);  primals_2 = primals_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:95 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_default_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_4, primals_3);  primals_4 = primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:97 in forward, code: left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        select_int: "i64[32, 512]" = torch.ops.aten.select.int(full_default_1, 2, 0)
        embedding_default_2: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_5, select_int)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:98 in forward, code: upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        select_int_1: "i64[32, 512]" = torch.ops.aten.select.int(full_default_1, 2, 1)
        embedding_default_3: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_6, select_int_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:99 in forward, code: right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        select_int_2: "i64[32, 512]" = torch.ops.aten.select.int(full_default_1, 2, 2)
        embedding_default_4: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_5, select_int_2);  primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:100 in forward, code: lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        select_int_3: "i64[32, 512]" = torch.ops.aten.select.int(full_default_1, 2, 3);  full_default_1 = None
        embedding_default_5: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_6, select_int_3);  primals_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:104 in forward, code: h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        sub_tensor: "i64[32, 512]" = torch.ops.aten.sub.Tensor(select_int_3, select_int_1);  select_int_3 = select_int_1 = None
        embedding_default_6: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_7, sub_tensor);  primals_7 = sub_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:105 in forward, code: w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        sub_tensor_1: "i64[32, 512]" = torch.ops.aten.sub.Tensor(select_int_2, select_int);  select_int_2 = select_int = None
        embedding_default_7: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_8, sub_tensor_1);  primals_8 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:106 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_default_8: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_9, full_default);  primals_9 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:109 in forward, code: words_embeddings
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None
        add_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_2);  add_tensor = embedding_default_2 = None
        add_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, embedding_default_3);  add_tensor_1 = embedding_default_3 = None
        add_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_2, embedding_default_4);  add_tensor_2 = embedding_default_4 = None
        add_tensor_4: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_3, embedding_default_5);  add_tensor_3 = embedding_default_5 = None
        add_tensor_5: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_4, embedding_default_6);  add_tensor_4 = embedding_default_6 = None
        add_tensor_6: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_5, embedding_default_7);  add_tensor_5 = embedding_default_7 = None
        add_tensor_7: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_6, embedding_default_8);  add_tensor_6 = embedding_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:119 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_7, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_8: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_8);  add_tensor_8 = None
        sub_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_tensor_7, getitem_1);  add_tensor_7 = getitem_1 = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default);  sub_tensor_2 = rsqrt_default = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_10);  mul_tensor = primals_10 = None
        add_tensor_9: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_11);  mul_tensor_1 = primals_11 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[25]" = torch.ops.prims.inductor_seeds.default(25, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:120 in forward, code: embeddings = self.dropout(embeddings)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, add_tensor_9);  gt_scalar = add_tensor_9 = None
        mul_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_0);  mul_tensor_3 = _shape_param_0 = None
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
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
