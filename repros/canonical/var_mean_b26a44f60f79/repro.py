"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_infer
Pattern hash: b26a44f60f79
Shape hash: 43e317b2
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
_shapes_config = "(T([50265, 768], f16), T([1, 4096], i64, gen=Index(50265)), T([1, 4096], i64), T([1, 4096], i32), T([4098, 768], f16), T([1, 768], f16), T([768], f16), T([768], f16))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f16[50265, 768]", arg0_1: "i64[1, 4096]", cumsum: "i64[1, 4096]", convert_element_type_1: "i32[1, 4096]", arg2_1: "f16[4098, 768]", arg3_1: "f16[1, 768]", arg4_1: "f16[768]", arg5_1: "f16[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1484 in forward, code: attention_mask = torch.ones(input_shape, device=device)
        full_default: "f32[1, 4096]" = torch.ops.aten.full.default([1, 4096], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_utils.py:998 in get_extended_attention_mask, code: extended_attention_mask = attention_mask[:, None, None, :]
        unsqueeze_default: "f32[1, 1, 4096]" = torch.ops.aten.unsqueeze.default(full_default, 1);  full_default = None
        unsqueeze_default_1: "f32[1, 1, 1, 4096]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_utils.py:1009 in get_extended_attention_mask, code: extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        convert_element_type_default: "f16[1, 1, 1, 4096]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_1, torch.float16);  unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_utils.py:1010 in get_extended_attention_mask, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        sub_tensor: "f16[1, 1, 1, 4096]" = torch.ops.aten.sub.Tensor(1.0, convert_element_type_default);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:418 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding_default: "f16[1, 4096, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:379 in create_position_ids_from_input_ids, code: incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
        convert_element_type_default_1: "i32[1, 4096]" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
        mul_tensor: "i32[1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_1);  convert_element_type_default_1 = convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:380 in create_position_ids_from_input_ids, code: return incremental_indices.long() + padding_idx
        convert_element_type_default_2: "i64[1, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.int64);  mul_tensor = None
        add_tensor: "i64[1, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_default_2, 1);  convert_element_type_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:419 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_default_1: "f16[1, 4096, 768]" = torch.ops.aten.embedding.default(arg2_1, add_tensor, 1);  arg2_1 = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:422 in forward, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        add_tensor_1: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1486 in forward, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        full_default_1: "i64[1, 4096]" = torch.ops.aten.full.default([1, 4096], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:420 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_default_2: "f16[1, 4096, 768]" = torch.ops.aten.embedding.default(arg3_1, full_default_1);  arg3_1 = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:422 in forward, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        add_tensor_2: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, embedding_default_2);  add_tensor_1 = embedding_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:423 in forward, code: embeddings = self.LayerNorm(embeddings)
        convert_element_type_default_3: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32);  add_tensor_2 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default_3, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 4096, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 4096, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor_1: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default_3, getitem_1);  convert_element_type_default_3 = getitem_1 = None
        add_tensor_3: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        mul_tensor_1: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default);  sub_tensor_1 = rsqrt_default = None
        mul_tensor_2: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg4_1);  mul_tensor_1 = arg4_1 = None
        add_tensor_4: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg5_1);  mul_tensor_2 = arg5_1 = None
        convert_element_type_default_4: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.float16);  add_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_utils.py:1010 in get_extended_attention_mask, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        full_default_2: "f16[1, 1, 1, 4096]" = torch.ops.aten.full.default([1, 1, 1, 4096], -0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1503 in forward, code: extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)[
        select_int: "f16[1, 1, 4096]" = torch.ops.aten.select.int(full_default_2, 1, 0);  full_default_2 = None
        select_int_1: "f16[1, 4096]" = torch.ops.aten.select.int(select_int, 1, 0);  select_int = None
        return (sub_tensor, convert_element_type_default_4, select_int_1)



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
